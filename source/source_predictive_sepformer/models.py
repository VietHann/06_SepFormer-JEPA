"""Training-only modules for source-predictive SepFormer."""

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
