"""JEPA Transformer blocks with length-independent sinusoidal positions."""

import math
from contextlib import nullcontext

import torch
from torch import nn
from speechbrain.lobes.models.dual_path import SBTransformerBlock
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding


class DynamicPositionalEncoding(PositionalEncoding):
    """Keep the standard position buffer, generating longer sequences on demand.

    The registered buffer never changes shape, so EMA, DDP buffer broadcasts
    and existing SBTransformerBlock state dictionaries retain their layout.
    """

    def forward(self, x):
        frames = x.size(1)
        if frames <= self.pe.size(1):
            return super().forward(x)
        channels = self.pe.size(-1)
        # Generate in FP32 like SpeechBrain's original sinusoidal table.
        positions = torch.arange(
            frames, device=x.device, dtype=torch.float32
        ).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, channels, 2, device=x.device, dtype=torch.float32)
            * -(math.log(10000.0) / channels)
        )
        angles = positions * denominator
        pe = torch.empty(frames, channels, device=x.device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe.unsqueeze(0).to(dtype=self.pe.dtype)


class MemoryEfficientAttention(nn.Module):
    """JEPA-only SpeechBrain attention adapter; never materialize CUDA weights.

    Reuse the original MHA child so state-dict keys and EMA parameter order
    stay unchanged. TransformerEncoder ignores the returned attention weights.
    """

    def __init__(self, attention):
        super().__init__()
        if not isinstance(attention, nn.MultiheadAttention) or attention.batch_first:
            raise ValueError("JEPA requires sequence-first PyTorch MultiheadAttention")
        self.att = attention

    def forward(
        self, query, key, value, attn_mask=None, key_padding_mask=None,
        return_attn_weights=True, pos_embs=None,
    ):
        if pos_embs is not None:
            attn_mask = pos_embs if attn_mask is None else attn_mask + pos_embs
        # batch_first=False prevents the native inference fast path (including
        # the frozen EMA target); need_weights=False routes MHA through SDPA.
        backend = (
            torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=True, enable_math=False,
            )
            if query.is_cuda else nullcontext()
        )
        try:
            with backend:
                output, _ = self.att(
                    query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1),
                    attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
        except RuntimeError as error:
            message = str(error).lower()
            if query.is_cuda and (
                "no available kernel" in message or "no viable backend" in message
            ):
                raise RuntimeError(
                    "JEPA requires memory-efficient CUDA SDPA; the math fallback "
                    "is disabled to avoid quadratic attention memory. "
                    f"Device={torch.cuda.get_device_name(query.device)}, "
                    f"dtype={query.dtype}, torch={torch.__version__}, "
                    f"CUDA={torch.version.cuda}. Run the CUDA preflight in "
                    "tests/test_transformer_blocks.py on this server and check "
                    "the installed PyTorch/CUDA support for this GPU and dtype. "
                    "No automatic precision change or windowing is performed."
                ) from error
            raise
        output = output.transpose(0, 1)
        return (output, None) if return_attn_weights else output


class JEPATransformerBlock(SBTransformerBlock):
    """JEPA Transformer with optional local windows and fused-only CUDA SDPA."""

    def __init__(self, num_layers, d_model, nhead, window_size=0, **kwargs):
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 0:
            raise ValueError("window_size must be a non-negative integer (0 means global)")
        self.window_size = window_size
        if kwargs.get("attention_type", "regularMHA") != "regularMHA":
            raise ValueError("JEPA memory-efficient attention requires regularMHA")
        super().__init__(num_layers, d_model, nhead, **kwargs)
        for layer in self.mdl.layers:
            layer.self_att = MemoryEfficientAttention(layer.self_att.att)
        if self.use_positional_encoding:
            self.pos_enc = DynamicPositionalEncoding(input_size=d_model)

    def forward(self, x):
        if not self.window_size or x.size(1) <= self.window_size:
            return super().forward(x)
        batch, frames, channels = x.shape
        complete = (frames // self.window_size) * self.window_size
        # Batch windows rather than launching one Transformer per window.
        # No padding: the exact tail is processed separately, so no artificial
        # frames influence context, predictor or the copied EMA target.
        windows = x[:, :complete].reshape(-1, self.window_size, channels)
        output = super().forward(windows).reshape(batch, complete, channels)
        if complete < frames:
            tail = super().forward(x[:, complete:])
            output = torch.cat((output, tail), dim=1)
        return output
