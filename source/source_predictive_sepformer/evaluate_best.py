#!/usr/bin/env python3
"""Evaluate the best saved checkpoint on Libri2Mix test data."""

import argparse
import json
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

import train as training_recipe
from metrics import mamba_tasnet_si_snri


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("hparams", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--verify-independent", action="store_true")
    return parser.parse_args()


def direct_si_snr(reference, estimate, eps=1e-8):
    """Compute SI-SNR without SpeechBrain's loss or PIT implementation."""
    reference = reference - reference.mean(dim=1, keepdim=True)
    estimate = estimate - estimate.mean(dim=1, keepdim=True)
    projection = (
        (estimate * reference).sum(dim=1, keepdim=True)
        * reference
        / (reference.square().sum(dim=1, keepdim=True) + eps)
    )
    noise = estimate - projection
    ratio = projection.square().sum(dim=1) / (
        noise.square().sum(dim=1) + eps
    )
    return 10.0 * torch.log10(ratio + eps)


def main():
    args = parse_args()
    with args.hparams.open(encoding="utf-8") as stream:
        hparams = load_hyperpyyaml(stream)

    # The CSV manifests already belong to this completed training run. Loading
    # them directly makes this command evaluation-only (no data prep or fit).
    _, _, test_data = training_recipe.baseline_recipe.dataio_prep(hparams)
    separator = training_recipe.SourcePredictiveSeparation(
        modules=hparams["modules"],
        opt_class=None,
        hparams=hparams,
        run_opts={"device": args.device},
        checkpointer=hparams["checkpointer"],
    )

    checkpoint = separator.checkpointer.recover_if_possible(min_key="si-snr")
    if checkpoint is None:
        raise RuntimeError("No checkpoint with metadata key 'si-snr' was found")

    # Explicitly freeze every module and switch off all training-time behavior.
    separator.modules.requires_grad_(False)
    separator.modules.eval()
    if any(module.training for module in separator.modules.values()):
        raise RuntimeError("At least one separator module is still in train mode")

    test_loader = separator.make_dataloader(
        test_data,
        stage=training_recipe.sb.Stage.TEST,
        batch_size=1,
        num_workers=0,
        ckpt_prefix=None,
    )

    sisnr_sum = 0.0
    baseline_sum = 0.0
    sisnri_sum = 0.0
    item_count = 0
    max_sisnr_difference = 0.0
    max_baseline_difference = 0.0
    max_sisnri_difference = 0.0
    with torch.inference_mode():
        for batch in tqdm(test_loader, dynamic_ncols=True, desc="test SI-SNRi"):
            targets = [batch.s1_sig, batch.s2_sig]
            if hparams["num_spks"] == 3:
                targets.append(batch.s3_sig)

            predictions, targets = separator.compute_forward(
                batch.mix_sig, targets, training_recipe.sb.Stage.TEST
            )
            mixture = batch.mix_sig[0].to(separator.device)
            mixture_sources = torch.stack(
                [mixture] * hparams["num_spks"], dim=-1
            )
            sisnr, baseline_sisnr, sisnri = mamba_tasnet_si_snri(
                hparams["loss"],
                predictions,
                targets,
                mixture,
                hparams["num_spks"],
            )

            if args.verify_independent:
                identity = direct_si_snr(targets, predictions).mean(dim=-1)
                swapped = direct_si_snr(
                    targets, predictions.flip(dims=(-1,))
                ).mean(dim=-1)
                direct_model = torch.maximum(identity, swapped)
                direct_baseline = direct_si_snr(
                    targets, mixture_sources
                ).mean(dim=-1)
                direct_improvement = direct_model - direct_baseline
                max_sisnr_difference = max(
                    max_sisnr_difference,
                    (sisnr - direct_model).abs().max().item(),
                )
                max_baseline_difference = max(
                    max_baseline_difference,
                    (baseline_sisnr - direct_baseline).abs().max().item(),
                )
                max_sisnri_difference = max(
                    max_sisnri_difference,
                    (sisnri - direct_improvement).abs().max().item(),
                )
            batch_items = sisnr.numel()
            sisnr_sum += sisnr.sum().item()
            baseline_sum += baseline_sisnr.sum().item()
            sisnri_sum += sisnri.sum().item()
            item_count += batch_items
            if args.max_items is not None and item_count >= args.max_items:
                break

    result = {
        "checkpoint": str(checkpoint.path),
        "checkpoint_metadata": checkpoint.meta,
        "metric": "Mamba-TasNet SI-SNRi",
        "eval": True,
        "frozen": all(
            not parameter.requires_grad
            for parameter in separator.modules.parameters()
        ),
        "test_items": item_count,
        "si_snr_db": sisnr_sum / item_count,
        "mixture_si_snr_db": baseline_sum / item_count,
        "si_snri_db": sisnri_sum / item_count,
    }
    if args.verify_independent:
        result["independent_check"] = {
            "max_si_snr_abs_difference_db": max_sisnr_difference,
            "max_mixture_si_snr_abs_difference_db": max_baseline_difference,
            "max_si_snri_abs_difference_db": max_sisnri_difference,
        }

    output = args.output
    if output is None:
        output = Path(hparams["output_folder"]) / "test_si_snri_best.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
