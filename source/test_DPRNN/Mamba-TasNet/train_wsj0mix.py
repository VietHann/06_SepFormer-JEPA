#!/usr/bin/env/python3

'''
Copied and modified from
https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/train.py
'''


"""Recipe for training a neural speech separation system on wsjmix the
dataset. The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer.yaml
> python train.py hparams/dualpath_rnn.yaml
> python train.py hparams/convtasnet.yaml

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both wsj2mix and
wsj3mix.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import os
import sys
import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import wandb
import logging
from speechbrain.core import AMPConfig

import pprint
pp = pprint.PrettyPrinter(indent=4)

from speechbrain.utils.logger import format_order_of_magnitude

os.environ['WANDB__SERVICE_WAIT'] = '999999'

def check_mean_max_grad(parameters):

    # Calculate gradient norm (L2 norm as an example)
    grad_norm = torch.sqrt(sum(p.grad.data.norm(2) ** 2 for p in parameters if p.grad is not None))

    # Calculate maximum gradient
    grad_max = max(p.grad.data.abs().max() for p in parameters if p.grad is not None)

    return grad_norm, grad_max


# Define training procedure
class Separation(sb.Brain):

    def _begin_training_diagnostics(self, ids):
        self._diagnostic_update_started = False
        if getattr(self.hparams, "gradient_diagnostics", False):
            if getattr(self, "_diagnostics", None) is None:
                from utils.training_diagnostics import TrainingDiagnostics
                self._diagnostics = TrainingDiagnostics(self.hparams)
            self._diagnostics.begin(ids)

    def _capture_training_diagnostics(self, mix, targets, mix_lens):
        diagnostics = getattr(self, "_diagnostics", None)
        if diagnostics is not None:
            diagnostics.capture(self, mix, targets, mix_lens)

    def _record_training_loss(self, name, loss):
        diagnostics = getattr(self, "_diagnostics", None)
        if diagnostics is not None:
            diagnostics.record_loss(name, loss)
        replay_terms = getattr(self, "_replay_loss_terms", None)
        if replay_terms is not None:
            replay_terms[name] = loss

    def _all_ranks_true(self, condition):
        """Keep numerical checks and skipped batches consistent under DDP."""
        flag = torch.tensor(int(bool(condition)), device=self.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        return bool(flag.item())

    def _require_finite(self, tensors, label):
        checks = [torch.isfinite(t.detach()).all().to(self.device) for t in tensors
                  if t is not None and (t.is_floating_point() or t.is_complex())]
        valid = bool(torch.stack(checks).all().item()) if checks else True
        if not self._all_ranks_true(valid):
            diagnostics = getattr(self, "_diagnostics", None)
            if diagnostics is not None:
                diagnostics.report(self, f"non-finite {label}", fatal=True)
            context = getattr(self, "_safety_context", "initialization")
            message = (
                f"Non-finite {label}; stage/batch={context}, step={self.step}. "
                "Stopping without saving or pruning checkpoints. "
                "An offending tensor may belong to another DDP rank."
            )
            logging.getLogger(__name__).error(message)
            raise FloatingPointError(message)

    def _clip_grad_norm(self, parameters):
        """Check the L2 norm before clipping, retrying overflow in float64."""
        gradients = [p.grad.detach() for p in parameters if p.grad is not None]
        norms = [torch.linalg.vector_norm(grad, 2) for grad in gradients]
        grad_norm = (torch.linalg.vector_norm(torch.stack(norms), 2) if norms
                     else torch.zeros((), device=self.device))
        use_float64 = not bool(torch.isfinite(grad_norm).item())
        if use_float64:
            norms = [torch.linalg.vector_norm(
                grad, 2, dtype=torch.complex128 if grad.is_complex() else torch.float64
            ) for grad in gradients]
            grad_norm = torch.linalg.vector_norm(torch.stack(norms), 2)
        self._require_finite([grad_norm], "gradient norm")
        diagnostics = getattr(self, "_diagnostics", None)
        if diagnostics is not None and not self._all_ranks_true(
            bool((grad_norm <= diagnostics.threshold).item())
        ):
            diagnostics.report(self, "large gradient norm before clipping", norm=grad_norm)
        if use_float64:
            logging.getLogger(__name__).warning(
                "Gradient norm overflow; recomputed in float64: norm=%.6e; "
                "stage/batch=%s, step=%s",
                grad_norm.item(), getattr(self, "_safety_context", "initialization"),
                self.step,
            )
        max_norm = self.hparams.clip_grad_norm
        if max_norm >= 0:
            clip_coef = (max_norm / (grad_norm + 1e-6)).clamp(max=1.0)
            for grad in gradients:
                if use_float64:
                    # Scale in float64 too, so small coefficients do not underflow.
                    dtype = torch.complex128 if grad.is_complex() else torch.float64
                    grad.copy_(grad.to(dtype=dtype).mul_(clip_coef))
                else:
                    grad.mul_(clip_coef)
        return grad_norm

    def _checked_optimizer_step(self, loss, use_amp=False):
        self._diagnostic_update_started = False
        self._record_training_loss("combined", loss)
        self._require_finite([loss], "training loss")
        if use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        parameters = [p for p in self.modules.parameters() if p.requires_grad]
        self._require_finite([p.grad for p in parameters], "gradients before clipping")
        self._clip_grad_norm(parameters)
        self._require_finite([p.grad for p in parameters], "gradients after clipping")
        self._diagnostic_update_started = True
        if use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self._require_finite(list(self.modules.parameters()), "weights after optimizer step")
        self._require_finite(
            [value for state in self.optimizer.state.values()
             for value in state.values() if torch.is_tensor(value)],
            "optimizer state after update",
        )
        self.update_jepa_target()
        self._require_finite(
            list(self.modules.parameters()) + list(self.modules.buffers()),
            "model/EMA state after update",
        )

    @property
    def jepa_enabled(self):
        return getattr(self.hparams, "jepa_weight", 0.0) > 0

    def add_jepa_objective(self, loss, waveform_loss, predictions, targets):
        """Add training-only JEPA, retaining the original SI-SNR threshold."""
        if not self.jepa_enabled:
            return loss
        from speechbrain.nnet.losses import PitWrapper, cal_si_snr

        latents = getattr(self, "_jepa_latents", None)
        self._jepa_latents = None
        if loss.nelement() == 0:
            return loss
        if latents is None:
            raise RuntimeError(
                "JEPA is enabled but no training latents were captured; "
                "compute_forward must run with stage=TRAIN before the JEPA loss."
            )
        if latents.ndim != 4 or latents.shape[:2] != targets.shape[::2]:
            raise ValueError(
                "JEPA latents must have shape [B,S,C,F] with the same batch/source "
                f"dimensions as targets; got {tuple(latents.shape)}"
            )
        pit = PitWrapper(lambda pred, target: cal_si_snr(target, pred))
        with torch.no_grad():
            # Solve PIT in waveform space, where the separator objective is
            # defined. Apply the same permutation to latent sources before
            # computing JEPA so speaker identities stay consistent.
            _, permutations = pit(predictions.detach().float(), targets.float())
        latent_for_pit = latents.permute(0, 2, 3, 1).contiguous()
        aligned = pit.reorder_tensor(latent_for_pit, permutations)
        aligned = aligned.permute(0, 3, 1, 2).contiguous()
        jepa_loss = self.modules.jepa(
            aligned, targets, self._jepa_valid_samples
        )
        self._record_training_loss("jepa_per_example", jepa_loss)
        self._require_finite([jepa_loss], "JEPA per-example loss")
        self.loss_stat["jepa-loss"] += jepa_loss.detach().sum().item()
        if self.hparams.threshold_byloss:
            eligible = waveform_loss > self.hparams.threshold
            if not eligible.any():
                return loss
            jepa_loss = jepa_loss[eligible]
        if jepa_loss.nelement() == 0:
            return loss
        jepa_term = self.hparams.jepa_weight * jepa_loss.mean()
        self._record_training_loss("jepa_weighted", jepa_term)
        return loss + jepa_term

    def update_jepa_target(self):
        if self.jepa_enabled:
            jepa = getattr(self.modules.jepa, "module", self.modules.jepa)
            jepa.update_target(self.hparams.Encoder)

    def train_augment(self, mix, targets, mix_lens):
        self._require_finite([mix, targets, mix_lens], "input audio or lengths")
        # Add speech distortions
        if self.jepa_enabled:
            sample_index = torch.arange(mix.shape[1], device=mix.device)
            valid = sample_index[None, :] < torch.round(mix_lens[:, None] * mix.shape[1])
            self._jepa_valid_samples = valid.unsqueeze(-1).expand_as(targets).clone()
        with torch.no_grad():
            if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                mix, targets = self.add_speed_perturb(targets, mix_lens)
                mix = targets.sum(-1)

            if self.hparams.use_wavedrop:
                mix = self.hparams.drop_chunk(mix, mix_lens)
                mix = self.hparams.drop_freq(mix)

            if self.hparams.limit_training_signal_len:
                mix, targets = self.cut_signals(mix, targets)

        return mix, targets

    def compute_forward(self, mix, stage):
        """Forward computations from the mixture to the separated signals."""
        # Separation
        mix_w = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask
        if stage == sb.Stage.TRAIN and self.jepa_enabled:
            self._jepa_latents = sep_h.permute(1, 0, 2, 3)

        # Decoding
        est_source = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :] # (B, T, S)

        return est_source

    def compute_objectives(self, predictions, targets, mix=None):
        """Computes the sinr loss"""
        loss = self.hparams.loss(targets, predictions)
        self._require_finite([predictions, targets, loss], "waveforms or SI-SNR loss")

        if mix != None:
            with torch.no_grad():
                mix = torch.stack(
                    [mix] * self.hparams.num_spks, dim=-1
                )
                sisnr_baseline = self.compute_objectives(
                    mix, targets
                )
                sisnr_i = loss - sisnr_baseline
                self._require_finite([sisnr_i], "SI-SNR improvement")
                self.loss_stat['-si-snr'] += loss.sum().item()
                self.loss_stat['-si-snri'] += sisnr_i.sum().item()
                self.count += loss.numel()

        return loss

    def fit_batch(self, batch):
        """Trains one batch"""
        self._safety_context = f"TRAIN ids={batch.id}"
        self._begin_training_diagnostics(batch.id)
        should_step = (self.step % self.grad_accumulation_factor) == 0

        # Unpacking batch list
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mixture
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)
    
        with self.no_sync(not should_step):
            if self.use_amp:
                with torch.autocast(
                    dtype=torch.bfloat16, device_type=torch.device(self.device).type,
                ):
                    mix, targets = self.train_augment(mix, targets, mix_lens)
                    self._capture_training_diagnostics(mix, targets, mix_lens)
                    predictions = self.compute_forward(mix, sb.Stage.TRAIN)

                    if hasattr(self.hparams, 'clip_wav_max'):
                        predictions = torch.clamp(predictions, 
                            min=-self.hparams.clip_wav_max, 
                            max=self.hparams.clip_wav_max
                        )

                    loss = self.compute_objectives(predictions, targets, mix)
                    waveform_loss = loss

                    # hard threshold the easy dataitems
                    if self.hparams.threshold_byloss:
                        th = self.hparams.threshold
                        loss = loss[loss > th]
                        if not self._all_ranks_true(loss.nelement() > 0):
                            self._jepa_latents = None
                            self.optimizer.zero_grad()
                            diagnostics = getattr(self, "_diagnostics", None)
                            if diagnostics is not None:
                                diagnostics.batch = None
                            return torch.tensor(0.0)
                        loss = loss.mean()
                    else:
                        loss = loss.mean()
                    self._record_training_loss("si_snr", loss)
                    loss = self.add_jepa_objective(loss, waveform_loss, predictions, targets)

                self._require_finite([loss], "combined training loss")
                if self._all_ranks_true(
                    loss.nelement() > 0 and loss < self.hparams.loss_upper_lim
                ):
                    self._checked_optimizer_step(loss, use_amp=True)

                    '''
                    grad_norm, grad_max = check_mean_max_grad(
                        self.optimizer.param_groups[0]['params']
                    )
                    est_max = (
                        torch.max(predictions[..., 0]) \
                        + torch.max(predictions[..., 1])
                    )/2
                    est_norm = (
                        torch.norm(predictions[..., 0], p=2) \
                        + torch.norm(predictions[..., 1], p=2)
                    )/2
                    tar_max = (
                        torch.max(targets[..., 0]) \
                        + torch.max(targets[..., 1])
                    )/2
                    tar_norm = (
                        torch.norm(targets[..., 0], p=2) \
                        + torch.norm(targets[..., 1], p=2)
                    )/2

                    self.hparams.train_logger.run.log(
                        data={
                            'train.grad_norm': float(grad_norm),
                            'train.grad_max': float(grad_max),
                            'train.step_loss': float(loss),
                            'train.est_max': float(est_max),
                            'train.est_norm': float(est_norm),
                            'train.est2tar_max': float(est_max/tar_max+1e-8),
                            'train.est2tar_norm': float(est_norm/tar_norm+1e-8),
                        },
                        step=self.global_fit_step
                    )
                    '''


                else:
                    self.nonfinite_count += 1
                    logger.info(
                        "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                            self.nonfinite_count
                        )
                    )
                    loss.data = torch.tensor(0).float().to(self.device)
            else:
                mix, targets = self.train_augment(mix, targets, mix_lens)
                self._capture_training_diagnostics(mix, targets, mix_lens)
                predictions = self.compute_forward(mix, sb.Stage.TRAIN)
                loss = self.compute_objectives(predictions, targets, mix)
                waveform_loss = loss

                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss = loss[loss > th]
                    if not self._all_ranks_true(loss.nelement() > 0):
                        self._jepa_latents = None
                        self.optimizer.zero_grad()
                        diagnostics = getattr(self, "_diagnostics", None)
                        if diagnostics is not None:
                            diagnostics.batch = None
                        return torch.tensor(0.0)
                    loss = loss.mean()
                else:
                    loss = loss.mean()
                self._record_training_loss("si_snr", loss)
                loss = self.add_jepa_objective(loss, waveform_loss, predictions, targets)

                self._require_finite([loss], "combined training loss")
                if self._all_ranks_true(
                    loss.nelement() > 0 and loss < self.hparams.loss_upper_lim
                ):
                    self._checked_optimizer_step(loss)
                else:
                    self.nonfinite_count += 1
                    logger.info(
                        "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                            self.nonfinite_count
                        )
                    )
                    loss.data = torch.tensor(0).float().to(self.device)

        self.optimizer.zero_grad()

        if hasattr(self, "warmup_scheduler") and \
            self.global_fit_step < self.hparams.n_warmup_step:
            self.warmup_scheduler.step()
        elif isinstance(
            self.hparams.lr_scheduler, CosineAnnealingLR
        ):
            self.hparams.lr_scheduler.step()
        self.global_fit_step += 1
        diagnostics = getattr(self, "_diagnostics", None)
        if diagnostics is not None:
            diagnostics.batch = None

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        self._safety_context = f"{stage.name} ids={batch.id}"
        result_writer = getattr(self, "_test_result_writer", None) if stage == sb.Stage.TEST else None
        if result_writer is not None:
            result_writer.start_batch()
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mixture
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)

        with torch.no_grad():
            predictions = self.compute_forward(mix, stage)
            loss = self.compute_objectives(predictions, targets, mix)
            if result_writer is not None:
                result_writer.write_batch(self, snt_id, predictions, targets, mix, mix_lens, loss)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.mean().detach()

    def on_fit_start(self):
        if self.grad_accumulation_factor != 1:
            raise ValueError("This recipe requires grad_accumulation_factor=1")
        self._compile()
        self._wrap_distributed()
        self.init_optimizers()

    @staticmethod
    def _cosine_resume_base_lr(requested_lr, eta_min, last_epoch, t_max):
        """Return cosine base LR that yields requested_lr at last_epoch."""
        import math

        factor = (1.0 + math.cos(math.pi * last_epoch / t_max)) / 2.0
        if requested_lr < eta_min:
            raise ValueError(
                f"resume_lr ({requested_lr}) must be at least cosine eta_min ({eta_min})"
            )
        if factor == 0:
            if requested_lr != eta_min:
                raise ValueError(
                    "Cannot resume above eta_min at the cosine schedule minimum"
                )
            return requested_lr
        return eta_min + (requested_lr - eta_min) / factor

    def _apply_resume_lr(self, requested_lr):
        """Set every optimizer group to the requested LR without restarting cosine."""
        import math

        requested_lr = float(requested_lr)
        if not math.isfinite(requested_lr) or requested_lr <= 0:
            raise ValueError("resume_lr must be a positive finite number")
        scheduler = self.hparams.lr_scheduler
        restored_lrs = [group["lr"] for group in self.optimizer.param_groups]
        cosine_position = None
        if isinstance(scheduler, CosineAnnealingLR):
            eta_min = scheduler.eta_min
            t_max = scheduler.T_max
            last_epoch = scheduler.last_epoch
            cosine_position = last_epoch
            base_lr = self._cosine_resume_base_lr(
                requested_lr, eta_min, last_epoch, t_max
            )
            scheduler.base_lrs = [base_lr for _ in self.optimizer.param_groups]
            for group in self.optimizer.param_groups:
                group["initial_lr"] = base_lr
        for group in self.optimizer.param_groups:
            group["lr"] = requested_lr
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = [requested_lr for _ in self.optimizer.param_groups]
        logging.getLogger(__name__).info(
            "Applying resumed learning-rate override: restored=%s requested=%s cosine_last_epoch=%s",
            restored_lrs, requested_lr, cosine_position,
        )
        # The explicit override supersedes any warmup schedule on recovery.
        if hasattr(self, "warmup_scheduler"):
            del self.warmup_scheduler

    def init_optimizers(self):

        if self.jepa_enabled:
            jepa = getattr(self.modules.jepa, "module", self.modules.jepa)
            jepa.initialize_target(self.hparams.Encoder)
        super().init_optimizers()
        # Load latest checkpoint to resume training if interrupted
        recovered_checkpoint = None
        if self.checkpointer is not None:
            recovered_checkpoint = self.checkpointer.recover_if_possible()

        self._require_finite(
            list(self.modules.parameters()) + list(self.modules.buffers()),
            "model state on initialization/resume",
        )
        self._require_finite(
            [value for state in self.optimizer.state.values()
             for value in state.values() if torch.is_tensor(value)],
            "optimizer state on resume",
        )
        cur_epoch = self.hparams.epoch_counter.current
        n_step_per_epoch = 20000 // self.hparams.batch_size
        cur_step = cur_epoch * n_step_per_epoch
        self.global_fit_step = cur_step
        print(f'Start training at {str(cur_step)} steps.')

        if hasattr(self.hparams, "n_warmup_step") and \
            self.hparams.n_warmup_step != 0: # Linear warmup
            from transformers import get_constant_schedule_with_warmup
            self.warmup_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.hparams.n_warmup_step
            )
            print(f'Linear warmup for {str(self.hparams.n_warmup_step)} steps.')

        if hasattr(self.hparams, "use_cosine_schedule") and self.hparams.use_cosine_schedule: # override lr scheduler
            self.hparams.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=n_step_per_epoch*self.hparams.N_epochs-self.hparams.n_warmup_step, 
                eta_min=self.hparams.lr*self.hparams.percent_lr,
                last_epoch=cur_step,
            )
            print('Use CosineAnnealingLR scheduler.')
            self.hparams.lr_scheduler._step_count = cur_step + 1
            self.optimizer._step_count = cur_step

        resume_lr = getattr(self.hparams, "resume_lr", None)
        if recovered_checkpoint is not None and resume_lr is not None:
            self._apply_resume_lr(resume_lr)
        if not (recovered_checkpoint is not None and resume_lr is not None):
            print(self.hparams.lr_scheduler)

    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        self._require_finite(
            list(self.modules.parameters()) + list(self.modules.buffers()),
            "checkpoint model state before evaluation",
        )

    def on_stage_start(self, stage, epoch=None):
        diagnostics = getattr(self, "_diagnostics", None)
        if diagnostics is not None:
            diagnostics.batch = None
        if sb.utils.distributed.if_main_process():
            print(f"[{stage.name}] Epoch {epoch}/{self.hparams.N_epochs}", flush=True)
        self.loss_stat = {
            'nan-ratio': 0,
            '-si-snr': 0,
            '-si-snri': 0,
        }
        self.count = 0
        if stage == sb.Stage.TRAIN and self.jepa_enabled:
            self.loss_stat["jepa-loss"] = 0.0

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        self._safety_context = f"{stage.name} epoch={epoch}"
        self._require_finite(
            [torch.tensor(float(stage_loss), device=self.device)], "epoch loss"
        )
        if not self._all_ranks_true(self.count > 0):
            raise RuntimeError("Empty stage: refusing to log metrics or save checkpoints")
        stage_stats = {"loss": round(stage_loss, 2)}
        for key in self.loss_stat.keys():
            stage_stats[key] = round(float(self.loss_stat[key]/self.count), 2)
        self._require_finite(
            [torch.tensor(list(stage_stats.values()), device=self.device)],
            "epoch metrics",
        )

        if stage == sb.Stage.TRAIN:
            self.train_stats = copy.deepcopy(stage_stats)
            print(f'Epoch {str(epoch)} TRAIN:')
            pp.pprint(stage_stats)

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # ReduceLROnPlateau
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)

            # IntervalScheduler
            elif isinstance(
                self.hparams.lr_scheduler, schedulers.IntervalScheduler
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(self.optimizer)
                print(self.optimizer.param_groups[0]["lr"])

            else:
                # no change
                current_lr = self.optimizer.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self._require_finite(
                list(self.modules.parameters()) + list(self.modules.buffers()),
                "model state before checkpoint save",
            )
            self._require_finite(
                [value for state in self.optimizer.state.values()
                 for value in state.values() if torch.is_tensor(value)],
                "optimizer state before checkpoint save",
            )
            self.checkpointer.save_and_keep_only(
                meta={"-si-snr": float(self.loss_stat["-si-snr"] / self.count),
                      "epoch": epoch},
                min_keys=["-si-snr"],
                num_to_keep=3,
                keep_recent=True,
                importance_keys=[],
            )

            print(f'Epoch {str(epoch)} VALID:')
            pp.pprint(stage_stats)

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

            print('TEST:')
            pp.pprint(stage_stats)


    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
            # Performing speed change (independently on each source)
            new_targets = []
            new_valid = []
            recombine = True

            for i in range(targets.shape[-1]):
                if self.hparams.use_speedperturb:
                    new_target = self.hparams.speed_perturb(targets[:, :, i])
                else:
                    # Random shifting must not implicitly enable speed
                    # perturbation when that augmentation is disabled.
                    new_target = targets[:, :, i]
                new_targets.append(new_target)
                if self.jepa_enabled:
                    valid = self._jepa_valid_samples[:, :, i].float().unsqueeze(1)
                    new_valid.append(F.interpolate(
                        valid, size=new_target.shape[-1], mode="nearest"
                    ).squeeze(1).bool())
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift,
                        self.hparams.max_shift + 1,
                        (1,),
                        device=targets.device,
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )
                    if self.jepa_enabled:
                        new_valid[i] = torch.roll(new_valid[i], shifts=(rand_shift[0],), dims=1)

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = targets.new_zeros(
                        targets.shape[0], min_len, targets.shape[-1]
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]
                if self.jepa_enabled:
                    self._jepa_valid_samples = torch.stack(
                        [valid[:, :min_len] for valid in new_valid], dim=-1
                    )

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length within the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        if self.jepa_enabled:
            self._jepa_valid_samples = self._jepa_valid_samples[
                :, randstart : randstart + self.hparams.training_signal_len, :
            ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data):
        """Restore the best checkpoint and stream all test metrics in one pass."""
        from utils.test_results import TestResultsWriter

        rank = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                rank = torch.distributed.get_rank()
        with TestResultsWriter(self.hparams.output_folder, rank=rank) as writer:
            self._test_result_writer = writer
            try:
                self.evaluate(test_data, min_key="-si-snr",
                              test_loader_kwargs={"batch_size": 1})
            finally:
                self._test_result_writer = None

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):

            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )


def dataio_prep(hparams):
    """Creates data processing pipeline"""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Provide audio pipelines

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        mix_sig = sb.dataio.dataio.read_audio(mix_wav)
        return mix_sig

    @sb.utils.data_pipeline.takes("s1_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        s1_sig = sb.dataio.dataio.read_audio(s1_wav)
        return s1_sig

    @sb.utils.data_pipeline.takes("s2_wav")
    @sb.utils.data_pipeline.provides("s2_sig")
    def audio_pipeline_s2(s2_wav):
        s2_sig = sb.dataio.dataio.read_audio(s2_wav)
        return s2_sig

    if hparams["num_spks"] == 3:

        @sb.utils.data_pipeline.takes("s3_wav")
        @sb.utils.data_pipeline.provides("s3_sig")
        def audio_pipeline_s3(s3_wav):
            s3_sig = sb.dataio.dataio.read_audio(s3_wav)
            return s3_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)
    if hparams["num_spks"] == 3:
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s3)
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig"]
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig"]
        )

    return train_data, valid_data, test_data


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if hparams.get("dataset") in ("librimix", "lrs2") and hparams["dynamic_mixing"]:
        raise ValueError(f"{hparams['dataset']} configuration requires dynamic_mixing=False; "
                         "the upstream dynamic mixer uses WSJ speaker IDs")

    # Check if wsj0_tr is set with dynamic mixing
    if hparams["dynamic_mixing"] and not os.path.exists(
        hparams["base_folder_dm"]
    ):
        raise ValueError(
            "Please, specify a valid base_folder_dm folder when using dynamic mixing"
        )

    # Data preparation; existing YAMLs retain the upstream WSJ0Mix path.
    from utils.prepare_data import prepare_wsjmix  # noqa

    prepare_data = prepare_wsjmix
    prepare_options = {}
    dataset = hparams.get("dataset", "wsj0mix")
    if dataset == "librimix":
        from utils.prepare_librimix import prepare_wsjmix as prepare_librimix
        prepare_data = prepare_librimix
    elif dataset == "lrs2":
        from utils.prepare_lrs2 import prepare_wsjmix as prepare_lrs2
        prepare_data = prepare_lrs2
    elif dataset != "wsj0mix":
        raise ValueError(f"Unsupported dataset: {dataset}")

    run_on_main(
        prepare_data,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "n_spks": hparams["num_spks"],
            "skip_prep": hparams["skip_prep"],
            "fs": hparams["sample_rate"],
            **prepare_options,
        },
    )

    # Create dataset objects
    if hparams["dynamic_mixing"]:
        from utils.dynamic_mixing import dynamic_mix_data_prep

        # if the base_folder for dm is not processed, preprocess them
        if "processed" not in hparams["base_folder_dm"]:
            # if the processed folder already exists we just use it otherwise we do the preprocessing
            if not os.path.exists(
                os.path.normpath(hparams["base_folder_dm"]) + "_processed"
            ):
                from utils.preprocess_dynamic_mixing import resample_folder

                print("Resampling the base folder")
                run_on_main(
                    resample_folder,
                    kwargs={
                        "input_folder": hparams["base_folder_dm"],
                        "output_folder": os.path.normpath(
                            hparams["base_folder_dm"]
                        )
                        + "_processed",
                        "fs": hparams["sample_rate"],
                        "regex": "**/*.wav",
                    },
                )
                # adjust the base_folder_dm path
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )
            else:
                print(
                    "Using the existing processed folder on the same directory as base_folder_dm"
                )
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )

        # Colleting the hparams for dynamic batching
        dm_hparams = {
            "train_data": hparams["train_data"],
            "data_folder": hparams["data_folder"],
            "base_folder_dm": hparams["base_folder_dm"],
            "sample_rate": hparams["sample_rate"],
            "num_spks": hparams["num_spks"],
            "training_signal_len": hparams["training_signal_len"],
            "dataloader_opts": hparams["dataloader_opts"],
        }
        train_data = dynamic_mix_data_prep(dm_hparams)
        _, valid_data, test_data = dataio_prep(hparams)
    else:
        train_data, valid_data, test_data = dataio_prep(hparams)

    # Initialize wandb

    if hparams['use_wandb']:
        hparams['train_logger'] = hparams['wandb_logger']()

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    # Training
    if not hparams['eval_only']:
        separator.fit(
            separator.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )

    # Eval
    separator.save_results(test_data)

    wandb.finish()
