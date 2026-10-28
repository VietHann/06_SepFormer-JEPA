"""Training-only masked source prediction inspired by nautel/jepaformer.

Uses its masked-latent / contextual EMA-target idea, not its training recipe
or separator. Sequence models are supplied by the Mamba-TasNet YAML.
Reference: https://github.com/nautel/jepaformer (85aef9049d12713d59756e242efd81284bcfdb1d).
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class MaskedSourceJEPA(nn.Module):
    """Predict clean-source embeddings from PIT-aligned, masked latents.

    The online waveform encoder remains owned by the original recipe. This
    module owns only the JEPA head, frozen target copies and EMA update count,
    so a single extra SpeechBrain recoverable saves the entire JEPA state.
    """

    def __init__(
        self, encoder, context_model, predictor_model, channels,
        predictor_dim=128, mask_ratio=0.5, mask_span=16,
        momentum=0.996, momentum_end=1.0, anneal_steps=4000000,
    ):
        super().__init__()
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be strictly between zero and one")
        if mask_span < 1 or int(mask_span) != mask_span:
            raise ValueError("mask_span must be a positive integer")
        if not 0 <= momentum <= momentum_end <= 1 or momentum == 1:
            raise ValueError("require 0 <= momentum < 1 and momentum <= momentum_end <= 1")
        if anneal_steps < 1:
            raise ValueError("anneal_steps must be positive")
        self.channels = channels
        self.mask_ratio = mask_ratio
        self.mask_span = int(mask_span)
        self.momentum = momentum
        self.momentum_end = momentum_end
        self.anneal_steps = anneal_steps
        self.context_model = context_model
        self.predictor_model = predictor_model
        self.input_proj = nn.Conv1d(channels, predictor_dim, 1)
        self.output_proj = nn.Conv1d(predictor_dim, channels, 1)
        self.mask_token = nn.Parameter(torch.empty(1, predictor_dim, 1))
        nn.init.normal_(self.mask_token, std=0.02)
        self.target_encoder = copy.deepcopy(encoder).requires_grad_(False)
        self.target_context = copy.deepcopy(context_model).requires_grad_(False)
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        self.target_context.eval()
        return self

    def _valid_frames(self, valid_samples, frames):
        """A frame is valid only if its complete receptive field is valid."""
        conv = self.target_encoder.conv1d

        def _one_dim(value, name):
            if isinstance(value, str):
                raise ValueError(
                    f"JEPA valid-frame propagation does not support {name}={value!r}"
                )
            if isinstance(value, tuple):
                if len(value) != 1:
                    raise ValueError(f"JEPA requires a 1-D convolution, got {name}={value}")
                return value[0]
            return value

        kernel = _one_dim(conv.kernel_size, "kernel_size")
        stride = _one_dim(conv.stride, "stride")
        padding = _one_dim(conv.padding, "padding")
        dilation = _one_dim(conv.dilation, "dilation")

        # Treat samples outside the waveform as invalid too. This matters when
        # the encoder uses non-zero Conv1d padding: a boundary frame must not
        # be used as a JEPA target merely because max_pool1d padded with zeros.
        invalid = (~valid_samples).float().unsqueeze(1)
        if padding:
            invalid = F.pad(invalid, (padding, padding), value=1.0)
        invalid = F.max_pool1d(
            invalid,
            kernel_size=kernel,
            stride=stride,
            padding=0,
            dilation=dilation,
        ).squeeze(1)
        if invalid.shape[-1] != frames:
            raise ValueError("JEPA requires the original Conv1d waveform encoder")
        return invalid == 0

    def _span_mask(self, valid):
        """Mask contiguous blocks; keep a visible frame when two are valid."""
        rows, frames = valid.shape
        if frames < 1:
            raise ValueError("JEPA requires at least one latent frame")
        span = min(self.mask_span, frames)
        blocks = (frames + span - 1) // span
        selected = torch.rand(rows, blocks, device=valid.device) < self.mask_ratio
        mask = selected.repeat_interleave(span, dim=1)[:, :frames] & valid
        # Guarantee supervision and visible context even on short examples.
        scores = torch.rand(rows, frames, device=valid.device).masked_fill(~valid, -1)
        first = scores.argmax(dim=1, keepdim=True)
        mask.scatter_(1, first, valid.any(dim=1, keepdim=True))
        scores.scatter_(1, first, -1)
        visible = scores.argmax(dim=1, keepdim=True)
        keep_visible = valid.sum(dim=1, keepdim=True) > 1
        mask.scatter_(1, visible, mask.gather(1, visible) & ~keep_visible)
        return mask & valid

    @staticmethod
    def _encode_valid_segments(model, features, valid):
        """Run [N,C,T] features without scanning padding in either direction.

        Invalid gaps may occur after source shifts, so each contiguous valid
        segment gets its own sequence state. Artificial JEPA masks remain
        inside those segments and are intentionally processed by the model.
        """
        if not valid.any():
            # Avoid running a stateful sequence model over an entirely padded
            # example. In addition to saving work, this prevents model biases
            # from creating meaningless hidden states in empty examples.
            return torch.zeros_like(features)
        if valid.all():
            output = model(features.transpose(1, 2)).transpose(1, 2)
            return output.masked_fill(~valid.unsqueeze(1), 0)
        edges = F.pad(valid.to(torch.int8), (1, 1)).diff(dim=1)
        starts = (edges == 1).nonzero().tolist()
        ends = (edges == -1).nonzero().tolist()
        output = torch.zeros_like(features)
        for (row, start), (_, end) in zip(starts, ends):
            segment = features[row : row + 1, :, start:end].transpose(1, 2)
            encoded = model(segment).transpose(1, 2)
            output[row : row + 1, :, start:end] = encoded.to(output.dtype)
        return output

    def forward(self, aligned_latents, clean_sources, valid_samples=None):
        """Return per-example loss for [B,S,C,F] latents and [B,T,S] audio."""
        if aligned_latents.ndim != 4 or clean_sources.ndim != 3:
            raise ValueError("expected latents [B,S,C,F] and clean sources [B,T,S]")
        batch, sources, channels, frames = aligned_latents.shape
        if batch < 1 or sources < 1 or frames < 1:
            raise ValueError("JEPA inputs must have non-empty batch, source, and frame axes")
        if channels != self.channels or clean_sources.shape[::2] != (batch, sources):
            raise ValueError("JEPA source/channel dimensions do not match")
        if clean_sources.shape[1] < 1:
            raise ValueError("clean_sources must contain at least one waveform sample")
        clean = clean_sources.transpose(1, 2).reshape(batch * sources, -1)
        if valid_samples is None:
            valid_samples = torch.ones_like(clean_sources, dtype=torch.bool)
        if valid_samples.shape != clean_sources.shape:
            raise ValueError("valid_samples must match clean sources [B,T,S]")
        valid_samples = valid_samples.to(dtype=torch.bool, device=clean_sources.device)
        valid_samples = valid_samples.transpose(1, 2).reshape(batch * sources, -1)
        valid = self._valid_frames(valid_samples, frames)
        mask = self._span_mask(valid)
        flat = aligned_latents.reshape(batch * sources, channels, frames)
        visible = flat.masked_fill(~(valid & ~mask).unsqueeze(1), 0)
        context = self._encode_valid_segments(self.context_model, visible, valid)
        h = self.input_proj(context)
        h = torch.where(mask.unsqueeze(1), self.mask_token.to(h.dtype), h)
        h = h.masked_fill(~valid.unsqueeze(1), 0)
        prediction = self._encode_valid_segments(self.predictor_model, h, valid)
        prediction = self.output_proj(prediction).float()
        with torch.no_grad():
            self.target_encoder.eval()
            self.target_context.eval()
            clean_latents = self.target_encoder(clean)
            if clean_latents.shape != (batch * sources, channels, frames):
                raise ValueError(
                    "target encoder output must match online latent shape: "
                    f"expected {(batch * sources, channels, frames)}, "
                    f"got {tuple(clean_latents.shape)}"
                )
            active = clean_latents.float().norm(dim=1) > 1e-4
            target = self._encode_valid_segments(
                self.target_context, clean_latents, valid
            ).transpose(1, 2)
            target = F.layer_norm(target.float(), (channels,)).transpose(1, 2)
        weights = (mask & valid & active).reshape(batch, sources, frames)
        frame_loss = (prediction - target).square().mean(dim=1)
        frame_loss = frame_loss.reshape(batch, sources, frames)
        return frame_loss.masked_fill(~weights, 0).sum(dim=(1, 2)) / weights.sum(
            dim=(1, 2)
        ).clamp_min(1)

    @torch.no_grad()
    def initialize_target(self, encoder):
        """Synchronize after recipe reinitialization, before checkpoint recovery."""
        encoder = getattr(encoder, "module", encoder)
        self.target_encoder.load_state_dict(encoder.state_dict())
        self.target_context.load_state_dict(self.context_model.state_dict())
        self.ema_updates.zero_()

    @torch.no_grad()
    def update_target(self, encoder):
        """Call once after a successful optimizer step, never on skipped steps."""
        encoder = getattr(encoder, "module", encoder)
        progress = min(1.0, self.ema_updates.item() / self.anneal_steps)
        momentum = self.momentum_end - (self.momentum_end - self.momentum) * (
            math.cos(math.pi * progress) + 1
        ) / 2
        for online, target in (
            (encoder, self.target_encoder), (self.context_model, self.target_context)
        ):
            for target_p, online_p in zip(target.parameters(), online.parameters()):
                target_p.lerp_(online_p.detach(), 1 - momentum)
            for target_b, online_b in zip(target.buffers(), online.buffers()):
                target_b.copy_(online_b)
        self.ema_updates.add_(1)
