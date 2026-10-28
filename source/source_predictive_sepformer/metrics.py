"""Evaluation metrics shared with the Mamba-TasNet recipe."""

import torch


@torch.no_grad()
def mamba_tasnet_si_snri(
    loss_fn,
    predictions,
    targets,
    mixture,
    num_spks,
):
    """Compute SI-SNR/SI-SNRi using Mamba-TasNet's exact convention.

    Mamba-TasNet stores SI-SNR as a negative PIT loss. Its SI-SNRi loss is
    ``output_loss - mixture_loss``; reported dB metrics negate those losses.
    """
    sisnr_loss = loss_fn(targets, predictions)
    mixture_estimates = torch.stack([mixture] * num_spks, dim=-1)
    sisnr_baseline_loss = loss_fn(targets, mixture_estimates)
    sisnri_loss = sisnr_loss - sisnr_baseline_loss

    finite = (
        torch.isfinite(sisnr_loss)
        & torch.isfinite(sisnr_baseline_loss)
        & torch.isfinite(sisnri_loss)
    )
    return (
        -sisnr_loss[finite],
        -sisnr_baseline_loss[finite],
        -sisnri_loss[finite],
    )
