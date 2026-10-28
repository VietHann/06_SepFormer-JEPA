"""Training-only modules for source-predictive SepFormer."""

import copy
import math
import os
from pathlib import Path
import warnings

import torch
from torch import nn
import torch.nn.functional as F


class SourceLatentPredictor(nn.Module):
    """Align separated SepFormer latents with clean-source targets.

    The predictor is pointwise in time so it cannot perform separation again
    using additional temporal context. It is discarded at inference time.
    """

    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, source_latents):
        """Predict clean-source features from ``[B, S, C, T]`` latents."""
        if source_latents.ndim != 4:
            raise ValueError(
                "source_latents must have shape [batch, sources, channels, time]"
            )

        batch, sources, channels, frames = source_latents.shape
        flattened = source_latents.reshape(batch * sources, channels, frames)
        predicted = self.network(flattened)
        return predicted.reshape(batch, sources, channels, frames)


class NormalizedSourcePredictionLoss(nn.Module):
    """Frame-level cosine distance for aligned source feature sequences."""

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, predictions, targets, valid_frames=None):
        """Return one source-prediction loss value per batch item.

        Arguments
        ---------
        predictions : torch.Tensor
            Predicted features shaped ``[B, S, C, T]``.
        targets : torch.Tensor
            Stop-gradient target features with the same shape.
        valid_frames : torch.Tensor, optional
            Number of valid latent frames shaped ``[B]``.
        """
        if predictions.shape != targets.shape:
            raise ValueError(
                "predictions and targets must have identical [B, S, C, T] shapes"
            )
        if predictions.ndim != 4:
            raise ValueError(
                "predictions and targets must have shape "
                "[batch, sources, channels, time]"
            )

        predictions = predictions.float()
        targets = targets.detach().float()
        target_norm = targets.norm(p=2, dim=2)
        predictions = F.normalize(predictions, p=2, dim=2, eps=self.eps)
        targets = F.normalize(targets, p=2, dim=2, eps=self.eps)
        frame_loss = 1.0 - (predictions * targets).sum(dim=2)
        active_mask = target_norm > self.eps

        frames = predictions.shape[-1]
        if valid_frames is None:
            valid_mask = active_mask
        else:
            if valid_frames.ndim != 1 or valid_frames.shape[0] != len(
                predictions
            ):
                raise ValueError("valid_frames must have shape [batch]")

            valid_frames = torch.clamp(
                valid_frames.to(device=predictions.device, dtype=torch.long),
                min=0,
                max=frames,
            )
            frame_index = torch.arange(frames, device=predictions.device)
            length_mask = frame_index.unsqueeze(0) < valid_frames.unsqueeze(1)
            valid_mask = active_mask & length_mask.unsqueeze(1)

        summed = (frame_loss * valid_mask).sum(dim=(1, 2))
        counts = valid_mask.sum(dim=(1, 2)).clamp_min(1)
        return summed / counts


# ---------------------------------------------------------------------------
# JEPA v2: Mamba sequence blocks, contextual EMA targets and masked prediction
# ---------------------------------------------------------------------------

try:
    from mamba_ssm import Mamba as _MambaSSM
except ImportError:  # pragma: no cover - depends on the CUDA extension
    _MambaSSM = None


class _TorchSelectiveSSM(nn.Module):
    """Slow reference selective SSM used only when ``mamba_ssm`` is missing.

    It follows the Mamba block layout (in-proj, depthwise conv, selective scan,
    gated out-proj) with a sequential scan, so it is suitable for smoke tests
    on CPU but not for real training.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        a = torch.arange(1, d_state + 1).float().repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        length = x.shape[1]
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :length].transpose(1, 2))
        dt, b, c = self.x_proj(x).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))
        a = -torch.exp(self.A_log.float())
        state = x.new_zeros(x.shape[0], x.shape[2], self.d_state)
        outputs = []
        for t in range(length):
            decay = torch.exp(dt[:, t, :, None] * a)
            state = decay * state + (dt[:, t, :, None] * b[:, t, None, :]) * x[
                :, t, :, None
            ]
            outputs.append((state * c[:, t, None, :]).sum(-1))
        y = torch.stack(outputs, dim=1) + x * self.D
        return self.out_proj(y * F.silu(z))


def _make_mamba(d_model, d_state, d_conv, expand):
    if _MambaSSM is not None:
        return _MambaSSM(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
    warnings.warn(
        "mamba_ssm not installed: using the slow PyTorch reference scan"
    )
    return _TorchSelectiveSSM(d_model, d_state, d_conv, expand)


class BiMambaLayer(nn.Module):
    """Pre-norm bidirectional Mamba layer with a feed-forward sublayer."""

    def __init__(self, d_model, d_ffn, d_state=16, d_conv=4, expand=2,
                 causal=False):
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(d_model)
        self.forward_ssm = _make_mamba(d_model, d_state, d_conv, expand)
        self.backward_ssm = (
            None if causal else _make_mamba(d_model, d_state, d_conv, expand)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_model)
        )

    def forward(self, x):
        h = self.norm1(x)
        y = self.forward_ssm(h)
        if self.backward_ssm is not None:
            y = y + self.backward_ssm(h.flip(1)).flip(1)
        x = x + y
        return x + self.ffn(self.norm2(x))


class BiMambaBlock(nn.Module):
    """Drop-in replacement for ``SBTransformerBlock`` ([B, L, N] -> same)."""

    def __init__(self, num_layers, d_model, d_ffn, d_state=16, d_conv=4,
                 expand=2, causal=False):
        super().__init__()
        self.layers = nn.ModuleList(
            BiMambaLayer(d_model, d_ffn, d_state, d_conv, expand, causal)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class SequenceContextEncoder(nn.Module):
    """Contextualize per-source latents ``[N, C, T]`` with a sequence model.

    The same module is used by the online branch (separated, span-masked
    latents) and, as an EMA copy, by the target branch (clean sources), so
    JEPA targets carry temporal context instead of raw filterbank features.
    """

    def __init__(self, sequence_model):
        super().__init__()
        self.sequence_model = sequence_model

    def forward(self, latents):
        return self.sequence_model(latents.transpose(1, 2)).transpose(1, 2)


class MaskedSpanPredictor(nn.Module):
    """Narrow predictor that infers target embeddings at masked frames.

    Unlike ``SourceLatentPredictor`` it sees temporal context, so the loss on
    masked spans can only be minimized by predicting, not by copying.
    """

    def __init__(self, channels, predictor_dim, sequence_model):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, predictor_dim, 1))
        self.input_proj = nn.Conv1d(channels, predictor_dim, 1)
        self.sequence_model = sequence_model
        self.output_proj = nn.Conv1d(predictor_dim, channels, 1)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, context, frame_mask):
        """``context`` [N, C, T]; ``frame_mask`` [N, T] (True = masked)."""
        h = self.input_proj(context)
        mask = frame_mask.unsqueeze(1).to(h.dtype)
        h = h * (1 - mask) + self.mask_token.to(h.dtype) * mask
        h = self.sequence_model(h.transpose(1, 2)).transpose(1, 2)
        return self.output_proj(h)


def sample_span_mask(batch, frames, mask_ratio, span, device):
    """Sample block masks covering about ``mask_ratio`` of ``frames``."""
    if mask_ratio <= 0:
        return torch.zeros(batch, frames, dtype=torch.bool, device=device)
    span = max(1, min(span, frames))
    num_spans = max(1, int(round(mask_ratio * frames / span)))
    starts = torch.randint(
        0, max(1, frames - span + 1), (batch, num_spans), device=device
    )
    offsets = torch.arange(span, device=device)
    index = (starts.unsqueeze(-1) + offsets).reshape(batch, -1)
    mask = torch.zeros(batch, frames, dtype=torch.bool, device=device)
    mask.scatter_(1, index, True)
    return mask


def ema_momentum(step, total_steps, start, end=1.0):
    """Cosine momentum schedule from ``start`` to ``end`` (I-JEPA/BYOL)."""
    if total_steps <= 0:
        return start
    progress = min(1.0, step / total_steps)
    return end - (end - start) * (math.cos(math.pi * progress) + 1) / 2


class MaskedLatentPredictionLoss(nn.Module):
    """Loss on masked frames between predictions and layer-normed targets.

    ``kind='l2'`` follows I-JEPA (MSE to LayerNorm-ed targets);
    ``kind='cosine'`` reuses the v1 frame-wise cosine distance;
    ``kind='infonce'`` is the contrastive alternative (CPC / pSkiM style),
    with the other masked frames of the batch as negatives.
    """

    def __init__(self, kind="l2", eps=1e-4, temperature=0.1):
        super().__init__()
        if kind not in ("l2", "cosine", "infonce"):
            raise ValueError("kind must be 'l2', 'cosine' or 'infonce'")
        self.kind = kind
        self.eps = eps
        self.temperature = temperature

    def forward(self, predictions, targets, weight_mask):
        """Shapes ``[B, S, C, T]`` and ``weight_mask`` ``[B, S, T]``."""
        predictions = predictions.float()
        targets = targets.detach().float()
        if self.kind == "infonce":
            return self._infonce(predictions, targets, weight_mask)
        if self.kind == "l2":
            targets = F.layer_norm(targets.transpose(2, 3), (targets.shape[2],))
            targets = targets.transpose(2, 3)
            frame_loss = (predictions - targets).pow(2).mean(dim=2)
        else:
            frame_loss = 1.0 - (
                F.normalize(predictions, dim=2, eps=self.eps)
                * F.normalize(targets, dim=2, eps=self.eps)
            ).sum(dim=2)
        weight_mask = weight_mask.float()
        summed = (frame_loss * weight_mask).sum(dim=(1, 2))
        return summed / weight_mask.sum(dim=(1, 2)).clamp_min(1)


    def _infonce(self, predictions, targets, weight_mask):
        """Frame-level InfoNCE inside each batch item (CPC-style baseline)."""
        batch, sources, channels, frames = predictions.shape
        query = F.normalize(
            predictions.permute(0, 1, 3, 2).reshape(batch, -1, channels),
            dim=-1,
            eps=self.eps,
        )
        key = F.normalize(
            targets.permute(0, 1, 3, 2).reshape(batch, -1, channels),
            dim=-1,
            eps=self.eps,
        )
        logits = torch.bmm(query, key.transpose(1, 2)) / self.temperature
        labels = torch.arange(logits.shape[1], device=logits.device)
        frame_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.repeat(batch),
            reduction="none",
        ).reshape(batch, sources, frames)
        weight_mask = weight_mask.float()
        return (frame_loss * weight_mask).sum(dim=(1, 2)) / weight_mask.sum(
            dim=(1, 2)
        ).clamp_min(1)


@torch.no_grad()
def embedding_health(embeddings, max_frames=4096):
    """Collapse monitors: mean per-channel std and effective rank.

    ``embeddings`` is ``[..., C, T]``; returns python floats.
    """
    x = embeddings.detach().float().transpose(-1, -2).reshape(
        -1, embeddings.shape[-2]
    )
    if x.shape[0] > max_frames:
        x = x[torch.randperm(x.shape[0], device=x.device)[:max_frames]]
    x = F.layer_norm(x, (x.shape[-1],))
    std = x.std(dim=0).mean().item()
    singular = torch.linalg.svdvals(x - x.mean(0))
    p = singular / singular.sum().clamp_min(1e-12)
    rank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()
    return std, rank


class _CausalConv1dLegacyAPI:
    """Adapt legacy Mamba low-level calls to causal-conv1d 1.4–1.6.

    Mamba-TasNet passes five/seven arguments; mamba-ssm 2.2.4 passes
    seven/ten. causal-conv1d 1.6 exposes allocation via cpp_functions.
    """

    def __init__(self, cuda_module, api_version):
        self._cuda = cuda_module
        self._api_version = api_version

    def causal_conv1d_fwd(self, x, weight, bias, seq_idx, *rest):
        if len(rest) == 1:
            initial_states, final_states_out, silu = None, None, rest[0]
        elif len(rest) == 3:
            initial_states, final_states_out, silu = rest
        else:
            raise TypeError("Unsupported causal_conv1d_fwd call signature")
        if self._api_version >= (1, 6):
            from causal_conv1d.cpp_functions import causal_conv1d_fwd_function

            return causal_conv1d_fwd_function(
                x, weight, bias, seq_idx, initial_states, final_states_out, silu
            )
        return self._cuda.causal_conv1d_fwd(
            x, weight, bias, seq_idx, initial_states, final_states_out, silu
        )

    def causal_conv1d_bwd(self, x, weight, bias, dout, seq_idx, *rest):
        if len(rest) == 2:
            initial_states, dfinal_states, dx, return_dinitial, silu = (
                None, None, rest[0], False, rest[1]
            )
            legacy = True
        elif len(rest) == 5:
            initial_states, dfinal_states, dx, return_dinitial, silu = rest
            legacy = False
        else:
            raise TypeError("Unsupported causal_conv1d_bwd call signature")
        if self._api_version >= (1, 6):
            from causal_conv1d.cpp_functions import causal_conv1d_bwd_function

            result = causal_conv1d_bwd_function(
                x, weight, bias, dout, seq_idx, initial_states,
                dfinal_states, dx, return_dinitial, silu,
            )
        else:
            result = self._cuda.causal_conv1d_bwd(
                x, weight, bias, dout, seq_idx, initial_states,
                dfinal_states, dx, return_dinitial, silu,
            )
        return result[:3] if legacy else result


def _patch_causal_conv1d_api():
    from importlib.metadata import version

    from modules.mamba import selective_scan_interface as ssi

    api_version = tuple(int(v) for v in version("causal-conv1d").split(".")[:2])
    if api_version >= (1, 4) and not isinstance(
        ssi.causal_conv1d_cuda, _CausalConv1dLegacyAPI
    ):
        ssi.causal_conv1d_cuda = _CausalConv1dLegacyAPI(
            ssi.causal_conv1d_cuda, api_version
        )
    if api_version >= (1, 6):
        from mamba_ssm.ops import selective_scan_interface as mamba_ssi

        if not isinstance(mamba_ssi.causal_conv1d_cuda, _CausalConv1dLegacyAPI):
            mamba_ssi.causal_conv1d_cuda = _CausalConv1dLegacyAPI(
                mamba_ssi.causal_conv1d_cuda, api_version
            )


def _resolve_mamba_tasnet_root():
    """Resolve the checkout that provides the official DPMamba blocks."""
    configured_root = os.environ.get("MAMBA_TASNET_ROOT")
    if configured_root:
        candidates = [Path(configured_root).expanduser().resolve()]
    else:
        project_root = Path(__file__).resolve().parent
        candidates = [
            project_root / "third_party" / "Mamba-TasNet",
            project_root.parent / "Mamba-TasNet",
        ]

    for root in candidates:
        if (root / "modules" / "mamba_blocks.py").is_file():
            return root

    searched = ", ".join(str(root) for root in candidates)
    raise ImportError(
        "Mamba-TasNet was not found. Set MAMBA_TASNET_ROOT to a checkout "
        f"containing modules/mamba_blocks.py. Searched: {searched}"
    )


def _official_dpmamba_block(num_layers, d_model, d_state):
    """Build the official bidirectional DPMamba block."""
    import sys

    root = _resolve_mamba_tasnet_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Mamba-TasNet imports ``mamba_ssm.ops.triton.layernorm`` (mamba-ssm 1.x);
    # mamba-ssm 2.x renamed it to ``layer_norm``. Alias it before importing the
    # official block so RMSNorm resolves correctly.
    try:
        import mamba_ssm.ops.triton.layernorm  # noqa: F401
    except ImportError:
        import mamba_ssm.ops.triton.layer_norm as _layer_norm

        sys.modules["mamba_ssm.ops.triton.layernorm"] = _layer_norm

    from modules.mamba_blocks import MambaBlocksSequential

    _patch_causal_conv1d_api()
    return MambaBlocksSequential(
        n_mamba=num_layers,
        bidirectional=True,
        d_model=d_model,
        d_state=d_state,
        expand=2,
        d_conv=4,
        fused_add_norm=False,
        rms_norm=True,
        residual_in_fp32=False,
    )


def build_sequence_model(kind, num_layers, d_model, d_ffn, nhead=8,
                         d_state=16, causal=False):
    """Build a ``[B, L, N]`` sequence model.

    ``kind``: ``'transformer'`` (SepFormer block), ``'mamba'`` (our BiMamba
    layers with FFN) or ``'dpmamba'`` (official DPMamba block).
    """
    if kind == "dpmamba":
        return _official_dpmamba_block(num_layers, d_model, d_state)
    if kind == "mamba":
        return BiMambaBlock(num_layers, d_model, d_ffn, d_state=d_state,
                            causal=causal)
    if kind == "transformer":
        from speechbrain.lobes.models.dual_path import SBTransformerBlock

        return SBTransformerBlock(
            num_layers=num_layers, d_model=d_model, nhead=nhead, d_ffn=d_ffn,
            dropout=0, use_positional_encoding=True, norm_before=True,
        )
    raise ValueError(f"unknown sequence model kind: {kind}")


def make_ema_copy(module):
    """Frozen deep copy used as an EMA target network."""
    target = copy.deepcopy(module)
    target.requires_grad_(False)
    return target.eval()
