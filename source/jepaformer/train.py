#!/usr/bin/env python3
"""Train SepFormer with source-predictive latent supervision on Libri2Mix."""

import csv
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEECHBRAIN_ROOT = Path(
    os.environ.get(
        "SPEECHBRAIN_ROOT",
        Path(__file__).resolve().parent / "third_party" / "speechbrain",
    )
)
BASE_RECIPE_DIR = SPEECHBRAIN_ROOT / "recipes" / "LibriMix" / "separation"

sys.path.insert(0, str(SPEECHBRAIN_ROOT))
sys.path.insert(0, str(BASE_RECIPE_DIR))

import speechbrain as sb  # noqa: E402
from speechbrain.nnet.losses import PitWrapper, cal_si_snr  # noqa: E402
from speechbrain.utils.distributed import run_on_main  # noqa: E402
from speechbrain.utils.logger import get_logger  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import embedding_health, ema_momentum, sample_span_mask  # noqa: E402


def _load_baseline_recipe():
    recipe_path = BASE_RECIPE_DIR / "train.py"
    spec = importlib.util.spec_from_file_location(
        "speechbrain_librimix_sepformer_recipe", recipe_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load SpeechBrain recipe from {recipe_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline_recipe = _load_baseline_recipe()
logger = get_logger(__name__)


def negative_si_snr(predictions, targets):
    """Return negative SI-SNR for prediction-target pairs."""
    return cal_si_snr(targets, predictions)


@sb.utils.checkpoints.register_checkpoint_hooks
class SourcePredictiveSeparation(baseline_recipe.Separation):
    """SepFormer training with PIT-aligned clean-source feature prediction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pit_si_snr = PitWrapper(negative_si_snr)
        momentum = self.hparams.target_encoder_momentum
        if not 0.0 <= momentum < 1.0:
            raise ValueError("target_encoder_momentum must be in [0, 1)")
        self.jepa_mode = self.hparams.jepa_mode
        if self.jepa_mode not in ("none", "pointwise", "masked"):
            raise ValueError("jepa_mode must be none, pointwise or masked")
        inactive = {
            "none": ("source_predictor", "target_encoder", "context_encoder",
                     "target_context_encoder", "span_predictor"),
            "pointwise": ("context_encoder", "target_context_encoder",
                          "span_predictor"),
            "masked": ("source_predictor",),
        }
        for name in inactive[self.jepa_mode]:
            self.modules[name].requires_grad_(False)
        self._jepa_ratio_ema = None
        self.nonfinite_count = 0
        self._freeze_target_encoder()
        self._reset_training_metrics()

    @sb.utils.checkpoints.mark_as_saver
    def _save(self, path):
        super()._save(path)
        with open(path, "a", encoding="utf-8") as stream:
            yaml.safe_dump({"jepa_ratio_ema": self._jepa_ratio_ema}, stream)

    @sb.utils.checkpoints.mark_as_loader
    def _recover(self, path, end_of_epoch):
        super()._recover(path, end_of_epoch)
        with open(path, encoding="utf-8") as stream:
            state = yaml.safe_load(stream)
        self._jepa_ratio_ema = state.get("jepa_ratio_ema")

    def _ema_pairs(self):
        """(online, target) module pairs updated by EMA."""
        pairs = [] if self.jepa_mode == "none" else [("encoder", "target_encoder")]
        if self.jepa_mode == "masked":
            pairs.append(("context_encoder", "target_context_encoder"))
        return pairs

    def _reset_training_metrics(self):
        self._source_loss_sum = 0.0
        self._total_loss_sum = 0.0
        self._metric_item_count = 0
        self._optimizer_steps = 0
        self._skipped_optimizer_steps = 0
        self._health = None
        self._last_jepa_weight = None

    def save_results(self, test_data):
        """Save one set of separation metrics for every test utterance.

        The upstream LibriMix recipe assumes a test batch contains exactly one
        item and calls ``Tensor.item()`` on the batch loss.  Our dataloader
        uses the configured batch size (two by default), so calculate and
        write each item in the batch separately instead.
        """
        from mir_eval.separation import bss_eval_sources

        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]
        all_sdrs, all_sdrs_i, all_sisnrs, all_sisnrs_i = [], [], [], []
        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )

        with open(save_file, "w", newline="", encoding="utf-8") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            with tqdm(test_loader, dynamic_ncols=True) as progress:
                for batch in progress:
                    mixture, mix_lengths = batch.mix_sig
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            batch.mix_sig, targets, sb.Stage.TEST
                        )

                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    ).to(targets.device)
                    valid_samples = (mix_lengths * mixture.shape[1]).round().long()
                    sisnr = self.compute_objectives(predictions, targets, valid_samples)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets, valid_samples
                    )
                    sisnr = sisnr.detach().reshape(-1)
                    sisnr_i = (sisnr - sisnr_baseline.detach().reshape(-1))

                    if sisnr.numel() != predictions.shape[0]:
                        raise RuntimeError(
                            "Expected one SI-SNR value per test item, got "
                            f"{sisnr.numel()} values for a batch of "
                            f"{predictions.shape[0]}."
                        )

                    for index, snt_id in enumerate(batch.id):
                        # Remove right-padding before mir_eval computes SDR.
                        signal_length = int(
                            round(mix_lengths[index].item() * mixture.shape[1])
                        )
                        reference = targets[index, :signal_length].t().cpu().numpy()
                        estimate = (
                            predictions[index, :signal_length]
                            .detach()
                            .cpu()
                            .numpy()
                            .T
                        )
                        baseline = (
                            mixture_signal[index, :signal_length]
                            .detach()
                            .cpu()
                            .numpy()
                            .T
                        )
                        sdr, _, _, _ = bss_eval_sources(reference, estimate)
                        sdr_baseline, _, _, _ = bss_eval_sources(
                            reference, baseline
                        )
                        sdr_value = float(sdr.mean())
                        sdr_i_value = float(sdr_value - sdr_baseline.mean())
                        sisnr_value = float(-sisnr[index].item())
                        sisnr_i_value = float(-sisnr_i[index].item())

                        writer.writerow(
                            {
                                "snt_id": snt_id,
                                "sdr": sdr_value,
                                "sdr_i": sdr_i_value,
                                "si-snr": sisnr_value,
                                "si-snr_i": sisnr_i_value,
                            }
                        )
                        all_sdrs.append(sdr_value)
                        all_sdrs_i.append(sdr_i_value)
                        all_sisnrs.append(sisnr_value)
                        all_sisnrs_i.append(sisnr_i_value)

            writer.writerow(
                {
                    "snt_id": "avg",
                    "sdr": np.mean(all_sdrs),
                    "sdr_i": np.mean(all_sdrs_i),
                    "si-snr": np.mean(all_sisnrs),
                    "si-snr_i": np.mean(all_sisnrs_i),
                }
            )

        logger.info("Mean SISNR is %s", np.mean(all_sisnrs))
        logger.info("Mean SISNRi is %s", np.mean(all_sisnrs_i))
        logger.info("Mean SDR is %s", np.mean(all_sdrs))
        logger.info("Mean SDRi is %s", np.mean(all_sdrs_i))

    def _pit_waveform_loss(self, predictions, targets, valid_samples=None):
        """Compute PIT on each utterance's real samples, excluding padding."""
        if valid_samples is None:
            return self.pit_si_snr(predictions, targets)
        lengths = valid_samples.detach().round().long().tolist()
        if len(lengths) != predictions.shape[0] or any(
            n <= 0 or n > predictions.shape[1] for n in lengths
        ):
            raise ValueError("valid_samples must contain one valid length per item")
        losses, permutations = [], []
        for index, length in enumerate(lengths):
            loss, permutation = self.pit_si_snr(
                predictions[index:index + 1, :length],
                targets[index:index + 1, :length],
            )
            losses.append(loss[0])
            permutations.append(permutation[0])
        return torch.stack(losses), permutations

    def compute_objectives(self, predictions, targets, valid_samples=None):
        return self._pit_waveform_loss(predictions, targets, valid_samples)[0]

    def _update_training_metrics(self, total_loss, source_loss):
        item_count = total_loss.numel()
        self._total_loss_sum += total_loss.detach().float().sum().item()
        self._source_loss_sum += source_loss.detach().float().sum().item()
        self._metric_item_count += item_count

    def _training_metric_stats(self):
        count = max(1, self._metric_item_count)
        return {
            "source-prediction-loss": self._source_loss_sum / count,
            "total-loss": self._total_loss_sum / count,
            "optimizer-steps": self._optimizer_steps,
            "skipped-optimizer-steps": self._skipped_optimizer_steps,
            **(self._health or {}),
            **(
                {"jepa-weight": self._last_jepa_weight}
                if self._last_jepa_weight is not None
                else {}
            ),
        }

    def on_stage_start(self, stage, epoch=None):
        super().on_stage_start(stage, epoch)
        if stage == sb.Stage.TRAIN:
            self._reset_training_metrics()
        else:
            self._sisnri_sum = 0.0
            self._sisnri_count = 0

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Log SI-SNRi for validation and test without changing selection."""
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            self.train_stats.update(self._training_metric_stats())
            return

        # ``compute_objectives`` is a negative SI-SNR loss, hence the
        # improvement is baseline_loss - estimated_loss.
        stage_stats["si-snri"] = self._sisnri_sum / max(1, self._sisnri_count)

        if stage == sb.Stage.VALID:
            if isinstance(
                self.hparams.lr_scheduler,
                baseline_recipe.schedulers.ReduceLROnPlateau,
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                baseline_recipe.schedulers.update_learning_rate(
                    self.optimizer, next_lr
                )
            else:
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            # Keep the existing SI-SNR criterion for checkpoint selection.
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"]
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def evaluate_batch(self, batch, stage):
        """Evaluate a batch and accumulate SI-SNR improvement per utterance."""
        if stage == sb.Stage.TRAIN:
            return super().evaluate_batch(batch, stage)

        mixture, mix_lengths = batch.mix_sig
        valid_samples = (mix_lengths * mixture.shape[1]).round().long()
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with torch.no_grad():
            predictions, targets = self.compute_forward(
                batch.mix_sig, targets, stage
            )
            loss = self.compute_objectives(predictions, targets, valid_samples)
            mixture_baseline = torch.stack(
                [mixture] * self.hparams.num_spks, dim=-1
            ).to(targets.device)
            baseline_loss = self.compute_objectives(
                mixture_baseline, targets, valid_samples
            )

        sisnri = (baseline_loss - loss).detach().reshape(-1)
        self._sisnri_sum += sisnri.float().sum().item()
        self._sisnri_count += sisnri.numel()

        # Preserve the base recipe's optional audio export behavior.
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(
                        batch.id[0], batch.mix_sig, targets, predictions
                    )
                    self.hparams.n_audio_to_save -= 1
            else:
                self.save_audio(batch.id[0], batch.mix_sig, targets, predictions)
        return loss.mean().detach()

    def _freeze_target_encoder(self):
        for _, target_name in self._ema_pairs():
            self.modules[target_name].requires_grad_(False)
            self.modules[target_name].eval()

    def initialize_target_encoder(self):
        """Initialize the frozen target networks from the online networks."""
        for online_name, target_name in self._ema_pairs():
            self.modules[target_name].load_state_dict(
                self.modules[online_name].state_dict()
            )
        self._freeze_target_encoder()

    def current_momentum(self):
        return ema_momentum(
            self.optimizer_step,
            self.hparams.ema_anneal_steps,
            self.hparams.target_encoder_momentum,
            self.hparams.target_encoder_momentum_end,
        )

    @torch.no_grad()
    def update_target_encoder(self):
        """Update the JEPA target networks from the online networks by EMA."""
        momentum = self.current_momentum()
        for online_name, target_name in self._ema_pairs():
            online = self.modules[online_name]
            target = self.modules[target_name]
            online = getattr(online, "module", online)
            target = getattr(target, "module", target)
            for target_parameter, online_parameter in zip(
                target.parameters(), online.parameters()
            ):
                target_parameter.mul_(momentum).add_(
                    online_parameter, alpha=1.0 - momentum
                )
            for target_buffer, online_buffer in zip(
                target.buffers(), online.buffers()
            ):
                target_buffer.copy_(online_buffer)
            target.eval()

    def _prepare_training_inputs(self, mix, targets, stage, noise=None):
        mix, mix_lens = mix
        mix = mix.to(self.device)
        valid_samples = (mix_lens.to(self.device) * mix.shape[1]).round().long()
        if (valid_samples <= 0).any() or (valid_samples > mix.shape[1]).any():
            raise ValueError("Mixture lengths must be positive and within the batch")
        targets = torch.stack(
            [targets[i][0] for i in range(self.hparams.num_spks)], dim=-1
        ).to(self.device)
        if stage != sb.Stage.TRAIN:
            return mix, valid_samples, targets

        mixtures, sources, lengths = [], [], []
        # Augment/crop real samples, not batch padding. Each source can have
        # a different speed; retain the exact common length after resampling.
        with torch.no_grad(), torch.autocast(device_type=mix.device.type, enabled=False):
            for index, length in enumerate(valid_samples.tolist()):
                source = targets[index, :length].float()
                mixture = mix[index, :length].float()
                if self.hparams.use_speedperturb:
                    resampled = [
                        self.hparams.speed_perturb(source[:, s].unsqueeze(0))[0]
                        for s in range(self.hparams.num_spks)
                    ]
                    length = min(signal.shape[0] for signal in resampled)
                    source = torch.stack([signal[:length] for signal in resampled], -1)
                if self.hparams.use_rand_shift:
                    source = torch.stack([
                        torch.roll(source[:, s], int(torch.randint(
                            self.hparams.min_shift, self.hparams.max_shift, (1,)
                        ).item()), dims=0)
                        for s in range(self.hparams.num_spks)
                    ], -1)
                if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                    mixture = source.sum(-1)
                    if self.hparams.use_wham_noise:
                        if noise is None:
                            raise ValueError("Noise is required with use_wham_noise")
                        length = min(length, noise.shape[1])
                        source = source[:length]
                        mixture = mixture[:length] + noise[index, :length].to(self.device)
                if self.hparams.use_wavedrop:
                    mixture = self.hparams.drop_chunk(
                        mixture.unsqueeze(0), torch.ones(1, device=mix.device)
                    )
                    mixture = self.hparams.drop_freq(mixture)[0]
                if self.hparams.limit_training_signal_len:
                    crop_length = self.hparams.training_signal_len
                    if crop_length <= 0:
                        raise ValueError("training_signal_len must be positive")
                    start = int(torch.randint(0, 1 + max(0, length - crop_length), (1,)).item())
                    mixture = mixture[start:start + crop_length]
                    source = source[start:start + crop_length]
                mixtures.append(mixture)
                sources.append(source)
                lengths.append(mixture.shape[0])

        mix = torch.nn.utils.rnn.pad_sequence(mixtures, batch_first=True)
        targets = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True)
        valid_samples = torch.tensor(lengths, device=mix.device, dtype=torch.long)
        return mix, valid_samples, targets

    def _valid_latent_frames(self, valid_samples, total_frames):
        encoder = self.modules.target_encoder.conv1d
        kernel = encoder.kernel_size[0]
        stride = encoder.stride[0]
        dilation = encoder.dilation[0]
        padding = encoder.padding[0]
        effective_kernel = dilation * (kernel - 1) + 1
        valid_frames = (
            torch.div(
                valid_samples + (2 * padding) - effective_kernel,
                stride,
                rounding_mode="floor",
            )
            + 1
        )
        return valid_frames.clamp(min=0, max=total_frames)

    def compute_forward_with_latents(self, mix, targets, stage, noise=None):
        """Separate a mixture and expose source and clean-target latents."""
        mix, valid_samples, targets = self._prepare_training_inputs(
            mix, targets, stage, noise
        )

        mixture_latent = self.modules.encoder(mix)
        estimated_masks = self.modules.masknet(mixture_latent)
        repeated_mixture = torch.stack([mixture_latent] * self.hparams.num_spks)
        separated_latents = repeated_mixture * estimated_masks

        estimated_sources = torch.cat(
            [
                self.modules.decoder(separated_latents[index]).unsqueeze(-1)
                for index in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        original_length = mix.size(1)
        estimated_length = estimated_sources.size(1)
        if original_length > estimated_length:
            estimated_sources = F.pad(
                estimated_sources,
                (0, 0, 0, original_length - estimated_length),
            )
        else:
            estimated_sources = estimated_sources[:, :original_length, :]

        clean_latents = None
        valid_frames = None
        if self.jepa_mode != "none":
            batch, samples, sources = targets.shape
            clean_waveforms = targets.permute(0, 2, 1).reshape(
                batch * sources, samples
            )
            with torch.no_grad():
                self.modules.target_encoder.eval()
                clean_latents = self.modules.target_encoder(clean_waveforms)
            clean_latents = clean_latents.reshape(
                batch,
                sources,
                clean_latents.shape[1],
                clean_latents.shape[2],
            )
            valid_frames = self._valid_latent_frames(
                valid_samples, clean_latents.shape[-1]
            )
        separated_latents = separated_latents.permute(1, 0, 2, 3)
        return (
            estimated_sources,
            targets,
            separated_latents,
            clean_latents,
            valid_frames,
            mix,
            valid_samples,
        )

    def compute_training_objectives(
        self,
        estimated_sources,
        targets,
        separated_latents,
        clean_latents,
        valid_frames,
        mixture,
        valid_samples=None,
    ):
        """Compute waveform PIT and source prediction with one assignment."""
        waveform_loss, permutations = self._pit_waveform_loss(
            estimated_sources, targets, valid_samples
        )
        if self.jepa_mode == "none":
            return waveform_loss, waveform_loss, torch.zeros_like(waveform_loss)

        latents_for_reordering = separated_latents.permute(0, 2, 3, 1)
        aligned_latents = self.pit_si_snr.reorder_tensor(
            latents_for_reordering, permutations
        ).permute(0, 3, 1, 2)
        if self.jepa_mode == "pointwise":
            predicted_clean_latents = self.modules.source_predictor(
                aligned_latents
            )
            source_prediction_loss = self.hparams.source_prediction_loss(
                predicted_clean_latents,
                clean_latents,
                valid_frames,
            )
        else:
            source_prediction_loss = self._masked_prediction_loss(
                aligned_latents, clean_latents, valid_frames, mixture
            )

        weight = self._jepa_weight(waveform_loss, source_prediction_loss)
        total_loss = waveform_loss + weight * source_prediction_loss
        return total_loss, waveform_loss, source_prediction_loss

    def _jepa_weight(self, waveform_loss, source_prediction_loss):
        """Fixed or gradient-balanced weight of the JEPA loss.

        With ``adaptive_jepa_weight`` the weight is chosen so that the JEPA
        gradient on the last MaskNet layer has ``source_prediction_weight``
        times the norm of the waveform gradient (adaptive weight as in VQGAN,
        Esser et al., CVPR 2021). Without it the raw JEPA gradient reaching
        the separator is ~1e-5..1e-3 of the SI-SNR gradient.
        """
        if not (
            self.hparams.adaptive_jepa_weight
            and self.modules.masknet.training
            and torch.is_grad_enabled()
        ):
            return self.hparams.source_prediction_weight

        interval = self.hparams.jepa_weight_interval
        if self._jepa_ratio_ema is not None and self.step % interval != 0:
            self._last_jepa_weight = (
                self.hparams.source_prediction_weight * self._jepa_ratio_ema
            )
            return self._last_jepa_weight

        masknet = getattr(self.modules.masknet, "module", self.modules.masknet)
        probe = masknet.end_conv1x1.weight
        scale = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
        wave_grad = torch.autograd.grad(
            waveform_loss.mean() * scale, probe, retain_graph=True
        )[0]
        jepa_grad = torch.autograd.grad(
            source_prediction_loss.mean() * scale, probe, retain_graph=True
        )[0]
        grad_energies = torch.stack((
            wave_grad.float().square().sum(), jepa_grad.float().square().sum()
        ))
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(grad_energies)
        ratio = grad_energies[0].sqrt() / (grad_energies[1].sqrt() + 1e-12)
        if torch.isfinite(ratio):
            ratio = ratio.clamp(max=self.hparams.max_jepa_weight_ratio).item()
            if self._jepa_ratio_ema is None:
                self._jepa_ratio_ema = ratio
            else:
                decay = 0.99 ** self.hparams.jepa_weight_interval
                self._jepa_ratio_ema = (
                    decay * self._jepa_ratio_ema + (1 - decay) * ratio
                )
        if self._jepa_ratio_ema is None:
            return self.hparams.source_prediction_weight
        self._last_jepa_weight = (
            self.hparams.source_prediction_weight * self._jepa_ratio_ema
        )
        return self._last_jepa_weight

    def _masked_prediction_loss(self, aligned_latents, clean_latents,
                                valid_frames, mixture):
        """JEPA v2: predict contextual embeddings of masked spans.

        Online: PIT-aligned separated latents -> span masking (frames zeroed)
        -> context encoder -> predictor with mask tokens.
        Target (``jepa_target``):
        ``clean``       EMA(encoder + context encoder) on the clean sources;
        ``mixture_sum`` EMA(...) on the mixture, matched by the sum of the
                        predicted source embeddings (needs no clean reference,
                        so the same objective works on unlabelled mixtures).
        """
        batch, sources, channels, frames = aligned_latents.shape
        flat = aligned_latents.reshape(batch * sources, channels, frames)
        # mixture_sum compares a per-item sum, so all sources share one mask
        frame_mask = sample_span_mask(
            batch if self.hparams.jepa_target == "mixture_sum" else batch * sources,
            frames,
            self.hparams.mask_ratio,
            self.hparams.mask_span,
            flat.device,
        )
        if self.hparams.jepa_target == "mixture_sum":
            frame_mask = frame_mask.repeat_interleave(sources, dim=0)
        visible = flat * (~frame_mask).unsqueeze(1).to(flat.dtype)
        context = self.modules.context_encoder(visible)
        predictions = self.modules.span_predictor(context, frame_mask)
        predictions = predictions.reshape(batch, sources, channels, frames)

        with torch.no_grad():
            self.modules.target_context_encoder.eval()
            if self.hparams.jepa_target == "mixture_sum":
                mixture_latent = self.modules.target_encoder(mixture)
                targets = self.modules.target_context_encoder(
                    mixture_latent
                ).unsqueeze(1)
            else:
                targets = self.modules.target_context_encoder(
                    clean_latents.reshape(batch * sources, channels, frames)
                ).reshape(batch, sources, channels, frames)

        if self.hparams.jepa_target == "mixture_sum":
            # sum of source predictions must reconstruct the mixture embedding
            predictions = predictions.sum(dim=1, keepdim=True)

        frame_index = torch.arange(frames, device=flat.device)
        valid = frame_index.unsqueeze(0) < valid_frames.to(flat.device).unsqueeze(1)
        active = clean_latents.detach().float().norm(dim=2) > 1e-4
        if self.hparams.jepa_target == "mixture_sum":
            active = active.any(dim=1, keepdim=True)
        weight = frame_mask.reshape(batch, sources, frames)[
            :, : targets.shape[1]
        ].float()
        weight = weight + self.hparams.visible_loss_weight * (1 - weight)
        weight = weight * (valid.unsqueeze(1) & active).float()

        if (
            self.step % self.hparams.health_interval == 0
            and torch.isfinite(predictions).all()
            and torch.isfinite(targets).all()
        ):
            pred_std, pred_rank = embedding_health(predictions)
            tgt_std, tgt_rank = embedding_health(targets)
            self._health = {
                "pred-std": pred_std,
                "pred-rank": pred_rank,
                "target-std": tgt_std,
                "target-rank": tgt_rank,
                "ema-momentum": self.current_momentum(),
            }
        return self.hparams.masked_prediction_loss(predictions, targets, weight)

    def _all_ranks_true(self, condition):
        flag = torch.tensor(int(bool(condition)), device=self.device, dtype=torch.int32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        return bool(flag.item())

    def _discard_forward(self, loss):
        # DDP still needs a backward pass to finish its pending reductions.
        if torch.distributed.is_initialized():
            safe_loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
            self.scaler.scale((safe_loss * 0.0).sum()).backward()
        self.optimizer.zero_grad()

    def fit_batch(self, batch):
        """Train one batch and exclude already-solved waveform examples."""
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        noise = batch.noise_sig[0] if self.hparams.use_wham_noise else None

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with self.training_ctx:
            forward_outputs = self.compute_forward_with_latents(
                mixture, targets, sb.Stage.TRAIN, noise
            )
            total_loss, waveform_loss, source_loss = (
                self.compute_training_objectives(*forward_outputs)
            )

            self.last_source_prediction_loss = source_loss.mean().detach()
            losses_are_finite = (
                torch.isfinite(waveform_loss).all()
                and torch.isfinite(source_loss).all()
                and torch.isfinite(total_loss).all()
            )
            if not self._all_ranks_true(losses_are_finite):
                self.nonfinite_count += 1
                self._skipped_optimizer_steps += 1
                logger.warning(
                    "Non-finite loss: waveform=%s source=%s total=%s "
                    "(%s batches this epoch)",
                    waveform_loss.detach().cpu().tolist(),
                    source_loss.detach().cpu().tolist(),
                    total_loss.detach().cpu().tolist(),
                    self.nonfinite_count,
                )
                self._discard_forward(total_loss)
                return torch.zeros(())

            full_total_loss = total_loss
            if self.hparams.threshold_byloss:
                selected = waveform_loss > self.hparams.threshold
                total_loss = total_loss[selected]
                waveform_loss = waveform_loss[selected]
                source_loss = source_loss[selected]

            if not self._all_ranks_true(total_loss.nelement() > 0):
                self._skipped_optimizer_steps += 1
                self._discard_forward(full_total_loss)
                return torch.zeros(())

            self._update_training_metrics(total_loss, source_loss)
            optimization_loss = total_loss.mean()
            reported_loss = waveform_loss.mean()

        if self._all_ranks_true(optimization_loss < self.hparams.loss_upper_lim):
            scale_before_step = self.scaler.get_scale()
            self.scaler.scale(optimization_loss).backward()
            self.scaler.unscale_(self.optimizer)
            max_norm = self.hparams.clip_grad_norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.modules.parameters(), max_norm if max_norm >= 0 else float("inf")
            )
            gradients_are_finite = torch.isfinite(grad_norm).to(dtype=torch.int32)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    gradients_are_finite, op=torch.distributed.ReduceOp.MIN
                )
            if not gradients_are_finite.item():
                self.nonfinite_count += 1
                self._skipped_optimizer_steps += 1
                if self.scaler.is_enabled():
                    self.scaler.update(
                        new_scale=scale_before_step * self.scaler.get_backoff_factor()
                    )
                logger.warning("Skipped non-finite gradients; optimizer and EMA unchanged")
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self._optimizer_steps += 1
                self.optimizer_step += 1
                self.update_target_encoder()
        else:
            self._skipped_optimizer_steps += 1
            self._discard_forward(full_total_loss)
            logger.warning(
                "Skipped excessive training loss: %s",
                optimization_loss.detach().item(),
            )

        self.optimizer.zero_grad()
        self.last_source_prediction_loss = source_loss.mean().detach()
        return reported_loss.detach().cpu()


def _load_libri_preparation():
    # Old external path:
    # /project/anhlt/lab-ss/LongHorn-TasNet-Libri/utils/prepare_data_libri.py
    # Resolve ./prepare_data_libri.py relative to this file, not the shell cwd.
    prep_path = Path(__file__).resolve().parent / "prepare_data_libri.py"
    spec = importlib.util.spec_from_file_location(
        "longhorn_librimix_preparation", prep_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load LibriMix preparation from {prep_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_lrs2_preparation():
    """Load the CSV preparation utilities for the LRS2 mixture dataset."""
    prep_path = Path(__file__).resolve().parent / "prepare_data_lrs2.py"
    spec = importlib.util.spec_from_file_location(
        "lrs2_preparation", prep_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load LRS2 preparation from {prep_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_datasets(hparams):
    # With skip_prep the CSVs already exist (e.g. from tools/make_libri2mix.py).
    if hparams["skip_prep"] and not hparams["dynamic_mixing"]:
        return baseline_recipe.dataio_prep(hparams)
    type_data = str(hparams.get("type_data", "libri2mix")).lower()

    if type_data in {"libri", "librimix", "libri2mix"}:
        data_preparation = _load_libri_preparation()
        preparation_kwargs = {
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "n_spks": hparams["num_spks"],
            "skip_prep": hparams["skip_prep"],
            "librimix_addnoise": hparams["use_wham_noise"],
            "fs": hparams["sample_rate"],
        }
    elif type_data == "lrs2":
        data_preparation = _load_lrs2_preparation()
        preparation_kwargs = {
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "n_spks": hparams["num_spks"],
            "skip_prep": hparams["skip_prep"],
        }
    else:
        raise ValueError(
            "Unsupported type_data={!r}. Expected one of: "
            "libri2mix, librimix, libri, lrs2.".format(type_data)
        )

    run_on_main(
        data_preparation.prepare_wsjmix,
        kwargs=preparation_kwargs,
    )

    if not hparams["dynamic_mixing"]:
        return baseline_recipe.dataio_prep(hparams)

    if type_data == "lrs2":
        raise ValueError(
            "dynamic_mixing is currently supported only for LibriMix "
            f"datasets, not {type_data!r}"
        )

    from dynamic_mixing import dynamic_mix_data_prep_librimix

    if "processed" not in hparams["base_folder_dm"]:
        processed_folder = (
            os.path.normpath(hparams["base_folder_dm"]) + "_processed"
        )
        if not os.path.exists(processed_folder):
            from recipes.LibriMix.meta.preprocess_dynamic_mixing import (
                resample_folder,
            )

            run_on_main(
                resample_folder,
                kwargs={
                    "input_folder": hparams["base_folder_dm"],
                    "output_folder": processed_folder,
                    "fs": hparams["sample_rate"],
                    "regex": "**/*.flac",
                },
            )
        hparams["base_folder_dm"] = processed_folder

    dynamic_hparams = {
        "train_data": hparams["train_data"],
        "data_folder": hparams["data_folder"],
        "base_folder_dm": hparams["base_folder_dm"],
        "sample_rate": hparams["sample_rate"],
        "num_spks": hparams["num_spks"],
        "training_signal_len": hparams["training_signal_len"],
        "dataloader_opts": hparams["dataloader_opts"],
    }
    train_data = dynamic_mix_data_prep_librimix(dynamic_hparams)
    _, valid_data, test_data = baseline_recipe.dataio_prep(hparams)
    return train_data, valid_data, test_data


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as stream:
        hparams = load_hyperpyyaml(stream, overrides)

    sb.utils.distributed.ddp_init_group(run_opts)
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if hparams["dynamic_mixing"] and not os.path.exists(
        hparams["base_folder_dm"]
    ):
        raise ValueError(
            "base_folder_dm must exist when dynamic_mixing is enabled"
        )

    if run_opts.get("device") == "cpu" and hparams.get("precision") == "fp16":
        hparams["precision"] = "bf16"

    train_data, valid_data, test_data = _prepare_datasets(hparams)

    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    separator = SourcePredictiveSeparation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Mamba layers rely on their own dt/A initialization, so they are not reset.
    if "pretrained_separator" not in hparams and hparams["seq_model"] not in (
        "mamba",
        "dpmamba",
    ):
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    separator.initialize_target_encoder()
    separator.fit(
        separator.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )
    separator.evaluate(test_data, min_key="si-snr")
    if not separator.debug:
        # Result export iterates over the full test set, so do it once after
        # distributed evaluation rather than letting every rank overwrite CSV.
        run_on_main(separator.save_results, args=[test_data])


if __name__ == "__main__":
    main()
