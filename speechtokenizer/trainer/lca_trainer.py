"""LCA (Low-load Channel-aware Adaptation) Trainer.

Derived from SpeechTokenizerTrainer. Adds:
  * Random L sampling per batch from {1, 2, 3}
  * ChannelSim (clean / index dropout / light substitution) on truncated codes
  * Communication-branch reconstruction loss alongside the full-depth path
  * Loading a Base checkpoint (generator weights only) for fine-tuning

The full-depth (n_q=N) GAN training path is preserved exactly as in the parent
trainer. LCA only adds an extra communication-branch forward + loss term, so
discriminators, distill loss, mel loss, etc. all keep working unchanged.
"""

from pathlib import Path
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .trainer import (
    SpeechTokenizerTrainer,
    accum_log,
    align_cosine_scheduler_to_training_plan,
    checkpoint_num_steps,
    exists,
    resolve_distill_loss_lambda,
)
from .loss import (
    adversarial_loss,
    discriminator_loss,
    feature_loss,
    mel_loss,
    mel_spectrogram,
    plot_spectrogram,
    recon_loss,
)


def apply_channel_sim_torch(codes: torch.Tensor, codebook_size: int, p_drop: float, p_sub: float, generator: torch.Generator):
    """ChannelSim implemented in pure torch on-device.

    Replicates scripts/channel_sim.apply_channel_sim semantics:
      * dropout uses previous-index replacement (t=0 keeps original)
      * substitution replaces with a uniform legal codebook index

    codes: (M, B, T) long tensor; M = current L (already truncated)
    Returns (perturbed codes, stats dict)
    """
    if codes.dtype != torch.long:
        raise ValueError(f"codes must be long tensor, got {codes.dtype}")
    if codes.dim() != 3:
        raise ValueError(f"expected codes shape (M, B, T), got {tuple(codes.shape)}")

    out = codes.clone()
    M, B, T = out.shape
    total = M * B * T
    replaced_by_previous = 0
    substituted = 0

    if p_drop > 0 and T > 1:
        u = torch.empty((M, B, T), device=out.device).uniform_(0.0, 1.0, generator=generator)
        drop_mask = u < p_drop
        # Don't drop t=0 (no previous index to replace with).
        drop_mask[:, :, 0] = False
        if drop_mask.any():
            prev = torch.cat([out[:, :, :1], out[:, :, :-1]], dim=2)
            out = torch.where(drop_mask, prev, out)
            replaced_by_previous = int(drop_mask.sum().item())

    if p_sub > 0:
        u = torch.empty((M, B, T), device=out.device).uniform_(0.0, 1.0, generator=generator)
        sub_mask = u < p_sub
        if sub_mask.any():
            replacement = torch.randint(0, codebook_size, (M, B, T), device=out.device, generator=generator, dtype=torch.long)
            out = torch.where(sub_mask, replacement, out)
            substituted = int(sub_mask.sum().item())

    if (out < 0).any() or (out >= codebook_size).any():
        raise RuntimeError("ChannelSim produced illegal codebook indices")

    stats = {
        "shape": list(out.shape),
        "L": M,
        "p_drop": p_drop,
        "p_sub": p_sub,
        "total_indices": total,
        "replaced_by_previous": replaced_by_previous,
        "substituted": substituted,
        "actual_p_drop": (replaced_by_previous / total) if total else 0.0,
        "actual_p_sub": (substituted / total) if total else 0.0,
    }
    return out, stats


class LCATrainer(SpeechTokenizerTrainer):
    """End-to-end LCA fine-tuning trainer."""

    def __init__(self, generator, discriminators, cfg, accelerate_kwargs=None):
        super().__init__(generator=generator, discriminators=discriminators, cfg=cfg, accelerate_kwargs=accelerate_kwargs or {})

        # LCA-specific config
        random_l_cfg = cfg.get("random_l_sampling", {})
        self.l_values = list(random_l_cfg.get("values", [1, 2, 3]))
        self.l_weights = random_l_cfg.get("weights")  # optional non-uniform sampling weights
        self.l_seed = int(random_l_cfg.get("seed", cfg.get("seed", 42)))
        self.l_strategy = random_l_cfg.get("strategy", "uniform_random")

        channel_cfg = cfg.get("channel_sim", {})
        self.channel_conditions = list(channel_cfg.get("conditions", [{"name": "clean", "p_drop": 0.0, "p_sub": 0.0}]))
        if not self.channel_conditions:
            raise ValueError("channel_sim.conditions must be a non-empty list")
        self.channel_strategy = channel_cfg.get("sample_strategy", "uniform_random")
        self.channel_weights = channel_cfg.get("weights")
        self.codebook_size = int(cfg.get("codebook_size", 1024))

        self.lambda_full = float(cfg.get("lambda_full", 1.0))
        self.lambda_comm = float(cfg.get("lambda_comm", 1.0))
        # Consistency loss between clean-decoded and perturbed-decoded outputs
        # of the same truncated codes. 0 disables (LCA v1 behavior).
        self.lambda_consistency = float(cfg.get("lambda_consistency", 0.0))

        # Independent generators for L / channel sampling so they don't perturb torch global RNG
        # used by the dataloader / dropout / cudnn.
        self._lca_py_rng = random.Random(self.l_seed)
        self._lca_torch_gen_cpu = torch.Generator(device="cpu")
        self._lca_torch_gen_cpu.manual_seed(self.l_seed + 7919)
        # device generator created lazily once we know the device
        self._lca_torch_gen_device = None

        # Optional: only train decoder (diagnostic ablation, not main route)
        self.train_mode = cfg.get("lca_train_mode", "end_to_end")
        if self.train_mode not in ("end_to_end", "decoder_only"):
            raise ValueError(f"lca_train_mode must be 'end_to_end' or 'decoder_only', got {self.train_mode!r}")

        if self.train_mode == "decoder_only":
            self._freeze_for_decoder_only()

        # JSONL training metrics file (separate from TensorBoard scalars)
        self.lca_metrics_path = Path(cfg["results_folder"]).parent / "metrics" / "lca_train_metrics.jsonl"
        self.lca_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._lca_metrics_fp = None

        # Provenance fields written by load_base_checkpoint
        self.base_checkpoint_path = None
        self.base_checkpoint_sha256 = None

        # ---- TB logging upgrade: running buckets for stratified comm metrics ----
        # Per-L training-side comm metric buffers (flushed every log_steps * 10 steps)
        self._comm_mel_by_L = {int(L): [] for L in self.l_values}
        self._comm_recon_by_L = {int(L): [] for L in self.l_values}
        # Per-channel comm metric buffers
        self._comm_mel_by_chan = {c["name"]: [] for c in self.channel_conditions}
        # Lifetime sampling counters (for L_dist / chan_dist running fractions)
        self._L_count = {int(L): 0 for L in self.l_values}
        self._chan_count = {c["name"]: 0 for c in self.channel_conditions}
        # Flush cadence for stratified buckets
        self._stratified_flush_every = max(1, int(cfg.get("log_steps", 100))) * 10
        # Sampling distribution log cadence
        self._sampling_dist_log_every = max(1, int(cfg.get("sampling_dist_log_every", 1000)))

        # ---- Dev-matrix evaluation: 3 L * 5 channels on a fixed subset ----
        self._dev_matrix_subset_size = int(cfg.get("dev_matrix_subset_size", 32))
        # Selected (L, channel_name) tuples for which we save dev audio + spec
        self._dev_audio_combos = cfg.get(
            "dev_audio_combos",
            [(1, "clean"), (1, "dropout-mid"), (3, "clean")],
        )
        # Normalize tuple-from-JSON (list pairs) to tuple form
        self._dev_audio_combos = [(int(L_), str(n)) for (L_, n) in self._dev_audio_combos]

        # ---- TB hparams text dump (runs once at __init__) ----
        if self.is_main:
            try:
                hparams_text = {
                    "exp3_experiment_tag": cfg.get("exp3_experiment_tag", ""),
                    "exp3_experiment_note": cfg.get("exp3_experiment_note", ""),
                    "lca_train_mode": self.train_mode,
                    "lambda_full": self.lambda_full,
                    "lambda_comm": self.lambda_comm,
                    "random_l_sampling": cfg.get("random_l_sampling"),
                    "channel_sim_conditions": [c.get("name") for c in self.channel_conditions],
                    "channel_sim_full": cfg.get("channel_sim"),
                    "n_q": cfg.get("n_q"),
                    "codebook_size": self.codebook_size,
                    "ideal_bitrate_per_L_bps": {int(L): int(L) * 50 * 10 for L in self.l_values},
                    "dev_matrix_subset_size": self._dev_matrix_subset_size,
                    "dev_audio_combos": self._dev_audio_combos,
                    "max_train_steps": cfg.get("max_train_steps"),
                    "total_train_steps": self.total_train_steps,
                    "lr_scheduler_total_steps": self.lr_scheduler_total_steps,
                }
                self.writer.add_text(
                    "hparams/lca_config",
                    "```json\n" + json.dumps(hparams_text, indent=2, ensure_ascii=False, default=str) + "\n```",
                    global_step=0,
                )
            except Exception as e:
                self.print(f"[LCA] hparams text dump failed: {e}")

    def _freeze_for_decoder_only(self):
        gen = self.accelerator.unwrap_model(self.generator)
        for name, p in gen.named_parameters():
            if name.startswith("encoder.") or name.startswith("quantizer.") or name.startswith("transform."):
                p.requires_grad = False
        if self.is_main:
            self.print("LCA decoder_only mode: encoder/quantizer/transform frozen")

    def _device_generator(self):
        if self._lca_torch_gen_device is None:
            self._lca_torch_gen_device = torch.Generator(device=self.device)
            self._lca_torch_gen_device.manual_seed(self.l_seed + 31337)
        return self._lca_torch_gen_device

    def _sample_L(self):
        if self.l_strategy == "round_robin":
            step = int(self.steps.item())
            return int(self.l_values[step % len(self.l_values)])
        if self.l_weights:
            return int(self._lca_py_rng.choices(self.l_values, weights=self.l_weights, k=1)[0])
        return int(self._lca_py_rng.choice(self.l_values))

    def _sample_channel(self):
        if self.channel_strategy == "round_robin":
            step = int(self.steps.item())
            return self.channel_conditions[step % len(self.channel_conditions)]
        if self.channel_weights:
            return self._lca_py_rng.choices(self.channel_conditions, weights=self.channel_weights, k=1)[0]
        return self._lca_py_rng.choice(self.channel_conditions)

    def load_base_checkpoint(self, base_checkpoint_path: str):
        """Load generator weights from an Exp2 SCIT-Speech-Base packaged checkpoint.

        The base checkpoint is a state_dict from torch.save(generator.state_dict()).
        We do NOT load discriminator/optimizer state — those start fresh for LCA.
        """
        path = Path(base_checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Base checkpoint not found: {path}")
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        gen = self.accelerator.unwrap_model(self.generator)
        missing, unexpected = gen.load_state_dict(state, strict=False)
        if self.is_main:
            self.print(f"LCA base ckpt loaded from {path}")
            if missing:
                self.print(f"  missing keys: {len(missing)} (showing first 5: {missing[:5]})")
            if unexpected:
                self.print(f"  unexpected keys: {len(unexpected)} (showing first 5: {unexpected[:5]})")

        import hashlib
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
        self.base_checkpoint_path = str(path)
        self.base_checkpoint_sha256 = sha.hexdigest()
        if self.is_main:
            self.print(f"  sha256: {self.base_checkpoint_sha256}")
            try:
                self.writer.add_text(
                    "hparams/base_checkpoint",
                    "```json\n" + json.dumps(
                        {
                            "base_checkpoint_path": self.base_checkpoint_path,
                            "base_checkpoint_sha256": self.base_checkpoint_sha256,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ) + "\n```",
                    global_step=0,
                )
            except Exception as e:
                self.print(f"[LCA] hparams/base_checkpoint dump failed: {e}")

    def save(self, path, best_dev_mel_loss):
        """Override save to also write SCIT-Speech-LCA_best.pt instead of SpeechTokenizer_best_dev.pt."""
        if best_dev_mel_loss < self.best_dev_mel_loss:
            self.best_dev_mel_loss = best_dev_mel_loss
            torch.save(
                self.accelerator.get_state_dict(self.generator),
                f"{self.results_folder}/SCIT-Speech-LCA_best.pt",
            )
            # Also keep a SpeechTokenizer_best_dev.pt copy so existing eval scripts still work
            torch.save(
                self.accelerator.get_state_dict(self.generator),
                f"{self.results_folder}/SpeechTokenizer_best_dev.pt",
            )
        ckpts = sorted(Path(path).parent.glob("SpeechTokenizerTrainer_*"))
        if len(ckpts) > self.num_ckpt_keep:
            for c in ckpts[: -self.num_ckpt_keep]:
                os.remove(c)
        pkg = dict(
            generator=self.accelerator.get_state_dict(self.generator),
            discriminators={k: self.accelerator.get_state_dict(v) for k, v in self.discriminators.items()},
            optim_g=self.optim_g.state_dict(),
            optim_d=self.optim_d.state_dict(),
            scheduler_g=self.scheduler_g.state_dict(),
            scheduler_d=self.scheduler_d.state_dict(),
            best_dev_mel_loss=self.best_dev_mel_loss,
            base_checkpoint_path=self.base_checkpoint_path,
            base_checkpoint_sha256=self.base_checkpoint_sha256,
            lca_py_rng_state=self._lca_py_rng.getstate(),
            lca_torch_gen_cpu_state=self._lca_torch_gen_cpu.get_state(),
            # Note: device generator state is intentionally NOT saved.
            # Serializing CUDA generator state through torch.save loses the byte
            # tensor type signature on reload, and ChannelSim's RNG drift across
            # restart only changes the local mask sequence, not the overall
            # uniform random distribution over training.
        )
        torch.save(pkg, path)

    def load(self, path=None, restore_optimizer=True):
        """Override load to also restore LCA-specific state (provenance + RNG)."""
        super().load(path=path, restore_optimizer=restore_optimizer)
        if not restore_optimizer:
            return
        # Re-read pkg to pick up LCA-specific fields the parent doesn't know about.
        if path is None:
            ckpts = sorted(self.results_folder.glob("SpeechTokenizerTrainer_*"))
            path = str(ckpts[-1])
        pkg = torch.load(path, map_location="cpu", weights_only=False)
        if pkg.get("base_checkpoint_path"):
            self.base_checkpoint_path = pkg["base_checkpoint_path"]
        if pkg.get("base_checkpoint_sha256"):
            self.base_checkpoint_sha256 = pkg["base_checkpoint_sha256"]
        if pkg.get("lca_py_rng_state") is not None:
            try:
                self._lca_py_rng.setstate(pkg["lca_py_rng_state"])
            except (TypeError, ValueError) as exc:
                if self.is_main:
                    self.print(f"[LCA] py_rng_state restore failed ({exc}); using fresh seed={self.l_seed}")
        if pkg.get("lca_torch_gen_cpu_state") is not None:
            try:
                self._lca_torch_gen_cpu.set_state(pkg["lca_torch_gen_cpu_state"])
            except RuntimeError as exc:
                if self.is_main:
                    self.print(f"[LCA] torch cpu gen state restore failed ({exc}); reseeding")
                self._lca_torch_gen_cpu.manual_seed(self.l_seed + 7919)
        # Device generator: not saved (see save() comment); reset to a deterministic
        # post-resume seed derived from current step so different resume points still
        # diverge predictably.
        self._lca_torch_gen_device = None
        if self.is_main:
            self.print(
                f"[LCA] continue_train state restored: base sha256={self.base_checkpoint_sha256}, "
                f"py_rng + torch_cpu RNG restored from pkg; device RNG re-initialized lazily"
            )

    def _open_lca_metrics(self):
        if self._lca_metrics_fp is None and self.is_main:
            self._lca_metrics_fp = open(self.lca_metrics_path, "a", encoding="utf-8")

    def _close_lca_metrics(self):
        if self._lca_metrics_fp is not None:
            try:
                self._lca_metrics_fp.flush()
                self._lca_metrics_fp.close()
            finally:
                self._lca_metrics_fp = None

    def train(self):
        """LCA training loop = parent loop + comm-branch forward + comm loss."""

        self.generator.train()
        for disc in self.discriminators.values():
            disc.train()
        if self.is_main:
            self._open_lca_metrics()

        step_time_log = {}

        steps = int(self.steps.item())
        if steps < self.num_warmup_steps:
            lr = self.warmup(steps)
            for pg in self.optim_g.param_groups:
                pg["lr"] = lr
            for pg in self.optim_d.param_groups:
                pg["lr"] = lr
        else:
            lr = self.scheduler_g.get_last_lr()[0]

        for epoch in range(self.epochs):
            if self.is_main:
                print(f"Epoch:{epoch} start...")

            for batch in self.dl:
                if steps >= self.total_train_steps:
                    self.print(f"planned_train_steps={self.total_train_steps} reached; stopping training")
                    self.print("training complete")
                    self._close_lca_metrics()
                    return
                if self.max_train_steps is not None and steps >= int(self.max_train_steps):
                    self.print(f"max_train_steps={self.max_train_steps} reached; stopping training early")
                    self.print("training complete")
                    self._close_lca_metrics()
                    return

                tic = time.time()
                x, semantic_feature = batch
                x = x.unsqueeze(1)

                # ============= Full-depth forward (parent loop) =============
                x_hat, loss_q, feature = self.generator(x)

                # discriminator step (against full-depth output, same as parent)
                self.optim_d.zero_grad(set_to_none=True)
                discriminator_outputs = [disc(x, x_hat.detach()) for disc in self.discriminators.values()]
                loss_disc_all = sum(discriminator_loss(*o[:2]) for o in discriminator_outputs)

                self.accelerator.backward(loss_disc_all)
                # ---- TB upgrade: discriminator grad norm at sync_gradients tick ----
                disc_grad_norm = None
                if self.accelerator.sync_gradients:
                    try:
                        sq = 0.0
                        for d in self.discriminators.values():
                            for p in d.parameters():
                                if p.grad is not None:
                                    sq += float(p.grad.detach().norm().item()) ** 2
                        disc_grad_norm = sq ** 0.5
                    except Exception:
                        disc_grad_norm = None
                    self.optim_d.step()
                    self.scheduler_d.step()
                    self.optim_d.zero_grad(set_to_none=True)

                # full-depth generator loss components
                discriminator_outputs = [disc(x, x_hat) for disc in self.discriminators.values()]
                loss_recon = recon_loss(x, x_hat)
                loss_mel_full = sum(
                    w * mel_loss(x, x_hat, **kw)
                    for (w, kw) in zip(self.mel_loss_lambdas, self.mel_loss_kwargs_list)
                )
                loss_feature_d = sum(feature_loss(*o[2:]) for o in discriminator_outputs)
                loss_adv = sum(adversarial_loss(o[1]) for o in discriminator_outputs)
                loss_distill = self.distill_loss(feature, semantic_feature)
                current_distill_lambda = resolve_distill_loss_lambda(self.cfg, steps)

                full_branch = (
                    loss_feature_d
                    + loss_adv
                    + loss_mel_full
                    + loss_q * self.commitment_loss_lambda
                    + loss_recon * self.recon_loss_lambda
                    + current_distill_lambda * loss_distill
                )

                # ============= Communication branch (LCA-specific) =============
                L = self._sample_L()
                cond = self._sample_channel()
                p_drop = float(cond.get("p_drop", 0.0))
                p_sub = float(cond.get("p_sub", 0.0))

                # Need codes for all M layers from the same encode pass, then truncate to L.
                # We re-run encode here (under no_grad for the encoder discrete codes themselves).
                # Then ChannelSim is applied on the L truncated codes, decoded back.
                # The decoder gradient still flows; in end_to_end mode encoder/quantizer also receive
                # gradient via the comm loss because we run a fresh forward that goes through them.
                #
                # Implementation: call generator.encode (no_grad) just to be safe about RVQ index
                # discreteness, then apply ChannelSim, then call generator.decode for gradient flow
                # on the decoder. For end_to_end, encoder/quantizer get gradient from the full-depth
                # branch above in the same backward pass.
                gen_unwrapped = self.accelerator.unwrap_model(self.generator)
                with torch.no_grad():
                    full_codes = gen_unwrapped.encode(x, n_q=int(self.cfg["n_q"]), st=0)
                    truncated = full_codes[:L].contiguous().long()
                pert_codes, sim_stats = apply_channel_sim_torch(
                    truncated,
                    codebook_size=self.codebook_size,
                    p_drop=p_drop,
                    p_sub=p_sub,
                    generator=self._device_generator(),
                )
                comm_out = gen_unwrapped.decode(pert_codes, st=0)
                loss_recon_comm = recon_loss(x, comm_out)
                loss_mel_comm = sum(
                    w * mel_loss(x, comm_out, **kw)
                    for (w, kw) in zip(self.mel_loss_lambdas, self.mel_loss_kwargs_list)
                )
                comm_branch = loss_mel_comm + loss_recon_comm * self.recon_loss_lambda

                # ---- LCA v2: consistency loss between clean and perturbed decode of same codes ----
                # Triggered only when (a) lambda_consistency > 0 AND (b) cond is not clean.
                # Compute clean decode of the same truncated codes and force its mel close to perturbed mel.
                # This gives an explicit "perturbation invariance" objective.
                loss_consistency = torch.tensor(0.0, device=x.device)
                used_consistency = False
                if self.lambda_consistency > 0.0 and (p_drop > 0.0 or p_sub > 0.0):
                    clean_codes = truncated  # same truncated codes, no ChannelSim applied
                    clean_out = gen_unwrapped.decode(clean_codes, st=0)
                    # Mel-domain L1 between clean and perturbed reconstructions.
                    # Use the same multi-scale mel kwargs list as full/comm branches for consistency.
                    loss_consistency = sum(
                        w * mel_loss(clean_out, comm_out, **kw)
                        for (w, kw) in zip(self.mel_loss_lambdas, self.mel_loss_kwargs_list)
                    )
                    comm_branch = comm_branch + self.lambda_consistency * loss_consistency
                    used_consistency = True

                # ============= Combined generator loss =============
                loss_generator_all = self.lambda_full * full_branch + self.lambda_comm * comm_branch

                self.accelerator.backward(loss_generator_all)
                # ---- TB upgrade: generator grad norm at sync_gradients tick ----
                gen_grad_norm = None
                if self.accelerator.sync_gradients:
                    try:
                        sq = 0.0
                        for p in self.generator.parameters():
                            if p.grad is not None:
                                sq += float(p.grad.detach().norm().item()) ** 2
                        gen_grad_norm = sq ** 0.5
                    except Exception:
                        gen_grad_norm = None
                    self.optim_g.step()
                    self.scheduler_g.step()
                    lr = self.scheduler_g.get_last_lr()[0]
                    self.optim_g.zero_grad(set_to_none=True)
                step_time_log = accum_log(step_time_log, {"time_cost": time.time() - tic})

                # ---- TB upgrade: NaN/Inf check on the combined loss ----
                nan_inf_flag = float(
                    torch.isnan(loss_generator_all).any().item()
                    or torch.isinf(loss_generator_all).any().item()
                )

                # ---- TB upgrade: per-L / per-channel comm metric buckets ----
                cur_mel_comm = float(loss_mel_comm.item())
                cur_recon_comm = float(loss_recon_comm.item())
                if L in self._comm_mel_by_L:
                    self._comm_mel_by_L[L].append(cur_mel_comm)
                    self._comm_recon_by_L[L].append(cur_recon_comm)
                if cond["name"] in self._comm_mel_by_chan:
                    self._comm_mel_by_chan[cond["name"]].append(cur_mel_comm)
                self._L_count[L] = self._L_count.get(L, 0) + 1
                self._chan_count[cond["name"]] = self._chan_count.get(cond["name"], 0) + 1

                # ============= Logging =============
                if self.is_main and not (steps % self.stdout_steps):
                    with torch.inference_mode():
                        mel_error = mel_loss(x, x_hat, **self.mel_loss_kwargs_list[0]).item()
                        mel_error_comm = mel_loss(x, comm_out, **self.mel_loss_kwargs_list[0]).item()
                    self.print(
                        f"Epoch {epoch} -- Step {steps}: GenAll: {loss_generator_all.item():0.3f}; "
                        f"FullMel: {mel_error:0.3f}; CommMel(L={L},{cond['name']}): {mel_error_comm:0.3f}; "
                        f"Q: {loss_q.item():0.3f}; Distill: {loss_distill.item():0.3f}; "
                        f"DLambda: {current_distill_lambda:0.3f}; "
                        f"t/step: {step_time_log['time_cost'] / self.stdout_steps:0.3f}s"
                    )
                    step_time_log = {}

                if self.is_main and not (steps % self.log_steps):
                    # mel_error / mel_error_comm are computed in the stdout block above
                    # only when steps % stdout_steps == 0. If log_steps is not aligned
                    # with stdout_steps, recompute defensively.
                    if steps % self.stdout_steps != 0:
                        with torch.inference_mode():
                            mel_error = mel_loss(x, x_hat, **self.mel_loss_kwargs_list[0]).item()
                            mel_error_comm = mel_loss(x, comm_out, **self.mel_loss_kwargs_list[0]).item()
                    log_payload = {
                        # parent-style scalars
                        "train/discriminators loss": loss_disc_all.item(),
                        "train/generator loss": loss_generator_all.item(),
                        "train/full_branch loss": float(full_branch.item()),
                        "train/comm_branch loss": float(comm_branch.item()),
                        "train/feature loss": loss_feature_d.item(),
                        "train/adversarial loss": loss_adv.item(),
                        "train/quantizer loss": loss_q.item(),
                        "train/mel loss full": loss_mel_full.item(),
                        "train/mel loss comm": loss_mel_comm.item(),
                        "train/mel error": mel_error,
                        "train/distillation loss": loss_distill.item(),
                        "train/distillation lambda": current_distill_lambda,
                        "train/learning_rate": lr,
                        "train/sampled_L": float(L),
                        "train/p_drop": p_drop,
                        "train/p_sub": p_sub,
                        # ---- TB upgrade #2: ChannelSim actual perturbation stats ----
                        "train/channel_sim/actual_p_drop": float(sim_stats.get("actual_p_drop", 0.0)),
                        "train/channel_sim/actual_p_sub": float(sim_stats.get("actual_p_sub", 0.0)),
                        "train/channel_sim/total_indices": float(sim_stats.get("total_indices", 0)),
                        "train/channel_sim/replaced_by_previous": float(sim_stats.get("replaced_by_previous", 0)),
                        "train/channel_sim/substituted": float(sim_stats.get("substituted", 0)),
                        # ---- TB upgrade #3: ideal bitrate ----
                        "train/ideal_bitrate_bps": float(L) * 50.0 * 10.0,
                        # ---- TB upgrade #4: comm branch component split ----
                        "train/loss_recon_comm": cur_recon_comm,
                        # ---- LCA v2: consistency loss (mel L1 between clean and perturbed decode) ----
                        "train/loss_consistency": float(loss_consistency.item()),
                        "train/lambda_consistency_used": float(self.lambda_consistency if used_consistency else 0.0),
                        # ---- TB upgrade #6: full vs comm ratio ----
                        "train/loss_ratio_comm_to_full": (
                            float(comm_branch.item()) / max(1e-6, float(full_branch.item()))
                        ),
                        # ---- TB upgrade #13: full-branch missing breakdowns ----
                        "train/full_branch/loss_recon": float(loss_recon.item()),
                        "train/full_branch/loss_distill_weighted": float(
                            (current_distill_lambda * loss_distill).item()
                        ),
                        # ---- TB upgrade #1: NaN/Inf flag (1.0 if degenerate, 0.0 otherwise) ----
                        "train/nan_inf_flag": nan_inf_flag,
                    }
                    if gen_grad_norm is not None:
                        log_payload["train/generator_grad_norm"] = float(gen_grad_norm)
                    if disc_grad_norm is not None:
                        log_payload["train/discriminator_grad_norm"] = float(disc_grad_norm)
                    self.log(log_payload, step=steps)

                # ---- TB upgrade #9 + #10: stratified per-L / per-channel comm metrics ----
                if (
                    self.is_main
                    and steps > 0
                    and (steps % self._stratified_flush_every == 0)
                ):
                    stratified_payload = {}
                    for Li, buf in self._comm_mel_by_L.items():
                        if buf:
                            stratified_payload[f"train/comm_mel/L{Li}"] = sum(buf) / len(buf)
                    for Li, buf in self._comm_recon_by_L.items():
                        if buf:
                            stratified_payload[f"train/comm_recon/L{Li}"] = sum(buf) / len(buf)
                    for name, buf in self._comm_mel_by_chan.items():
                        if buf:
                            stratified_payload[f"train/comm_mel/channel_{name}"] = sum(buf) / len(buf)
                    if stratified_payload:
                        self.log(stratified_payload, step=steps)
                    for buf in self._comm_mel_by_L.values():
                        buf.clear()
                    for buf in self._comm_recon_by_L.values():
                        buf.clear()
                    for buf in self._comm_mel_by_chan.values():
                        buf.clear()

                # ---- TB upgrade #11: L / channel sampling distribution sanity ----
                if (
                    self.is_main
                    and steps > 0
                    and (steps % self._sampling_dist_log_every == 0)
                ):
                    tot_L = sum(self._L_count.values()) or 1
                    tot_c = sum(self._chan_count.values()) or 1
                    self.log(
                        {f"train/L_dist/L{Li}": v / tot_L for Li, v in self._L_count.items()},
                        step=steps,
                    )
                    self.log(
                        {f"train/chan_dist/{name}": v / tot_c for name, v in self._chan_count.items()},
                        step=steps,
                    )

                # JSONL detailed record (one per logged step)
                if self.is_main and not (steps % self.log_steps) and self._lca_metrics_fp is not None:
                    record = {
                        "step": steps,
                        "epoch": epoch,
                        "sampled_L": L,
                        "channel": cond.get("name", "unknown"),
                        "p_drop": p_drop,
                        "p_sub": p_sub,
                        "full_branch": float(full_branch.item()),
                        "comm_branch": float(comm_branch.item()),
                        "loss_q": float(loss_q.item()),
                        "loss_distill": float(loss_distill.item()),
                        "loss_recon_comm": float(loss_recon_comm.item()),
                        "loss_mel_comm": float(loss_mel_comm.item()),
                        "loss_consistency": float(loss_consistency.item()),
                        "consistency_used": bool(used_consistency),
                        "lr": float(lr),
                        "channel_stats": sim_stats,
                    }
                    self._lca_metrics_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    self._lca_metrics_fp.flush()

                self.accelerator.wait_for_everyone()

                # ============= Validate + save =============
                if self.is_main and not (steps % self.save_model_steps) and steps != 0:
                    self.print("Validation start ...")
                    total_mel_error = 0.0
                    total_distill_loss = 0.0
                    total_q_loss = 0.0
                    num = 0
                    self.generator.eval()
                    # Cache fixed dev subset for comm-matrix evaluation (first N batches)
                    dev_subset_xv = []
                    with torch.inference_mode():
                        for i, vbatch in tqdm(enumerate(self.valid_dl)):
                            xv, semantic_v = vbatch
                            xv = xv.unsqueeze(1)
                            xv_hat, vq, vf = self.generator(xv)
                            mel_e = mel_loss(xv, xv_hat, **self.mel_loss_kwargs_list[0]).item()
                            total_mel_error += mel_e
                            total_distill_loss += self.distill_loss(vf, semantic_v).item()
                            total_q_loss += float(vq.item())
                            num += xv.size(0)
                            if len(dev_subset_xv) < self._dev_matrix_subset_size:
                                dev_subset_xv.append(xv.cpu())
                            if i < self.showpiece_num:
                                if not self.plot_gt_once:
                                    self.log(
                                        {f"groundtruth/x_{i}": xv[0].cpu().detach()},
                                        type="audio",
                                        sample_rate=self.sample_rate,
                                        step=steps,
                                    )
                                    x_spec = mel_spectrogram(xv.squeeze(1), **self.mel_kwargs)
                                    self.log(
                                        {f"groundtruth/x_spec_{i}": plot_spectrogram(x_spec[0].cpu().numpy())},
                                        type="figure",
                                        step=steps,
                                    )
                                self.log(
                                    {f"generate/x_hat_{i}": xv_hat[0].cpu().detach()},
                                    type="audio",
                                    sample_rate=self.sample_rate,
                                    step=steps,
                                )
                                xh_spec = mel_spectrogram(xv_hat.squeeze(1), **self.mel_kwargs)
                                self.log(
                                    {f"generate/x_hat_spec_{i}": plot_spectrogram(xh_spec[0].cpu().numpy())},
                                    type="figure",
                                    step=steps,
                                )
                        if not self.plot_gt_once:
                            self.plot_gt_once = True
                        avg_mel = total_mel_error / max(1, num)
                        avg_distill = total_distill_loss / max(1, num)
                        avg_q = total_q_loss / max(1, num)
                        self.print(
                            f"{steps}: dev mel error: {avg_mel:0.3f}\t"
                            f"dev distill loss: {avg_distill:0.3f}\t"
                            f"dev quantizer loss: {avg_q:0.3f}"
                        )
                        # ---- TB upgrade #8: keep legacy "dev/mel error" tag for back-compat,
                        # add explicit "dev/full_depth_mel_error" + "dev/quantizer_loss" ----
                        self.log(
                            {
                                "dev/mel error": avg_mel,  # legacy alias
                                "dev/full_depth_mel_error": avg_mel,
                                "dev/distillation loss": avg_distill,
                                "dev/quantizer_loss": avg_q,
                            },
                            step=steps,
                        )

                        # ---- TB upgrade #5: dev-matrix comm evaluation ----
                        # 3 L * 5 channels on a fixed subset, mel + recon
                        dev_matrix_payload = {}
                        gen_unwrapped_eval = self.accelerator.unwrap_model(self.generator)
                        for L_eval in self.l_values:
                            for cond_eval in self.channel_conditions:
                                mel_acc, rec_acc, n_acc = 0.0, 0.0, 0
                                for xv_cpu in dev_subset_xv:
                                    xv_dev = xv_cpu.to(self.device)
                                    full_codes_dev = gen_unwrapped_eval.encode(
                                        xv_dev, n_q=int(self.cfg["n_q"]), st=0
                                    )
                                    truncated_dev = full_codes_dev[:int(L_eval)].contiguous().long()
                                    pert_dev, _ = apply_channel_sim_torch(
                                        truncated_dev,
                                        codebook_size=self.codebook_size,
                                        p_drop=float(cond_eval.get("p_drop", 0.0)),
                                        p_sub=float(cond_eval.get("p_sub", 0.0)),
                                        generator=self._device_generator(),
                                    )
                                    yhat_dev = gen_unwrapped_eval.decode(pert_dev, st=0)
                                    mel_acc += mel_loss(
                                        xv_dev, yhat_dev, **self.mel_loss_kwargs_list[0]
                                    ).item()
                                    rec_acc += recon_loss(xv_dev, yhat_dev).item()
                                    n_acc += 1
                                if n_acc > 0:
                                    name = cond_eval["name"]
                                    dev_matrix_payload[
                                        f"dev/comm_mel/L{int(L_eval)}_{name}"
                                    ] = mel_acc / n_acc
                                    dev_matrix_payload[
                                        f"dev/comm_recon/L{int(L_eval)}_{name}"
                                    ] = rec_acc / n_acc
                        if dev_matrix_payload:
                            self.log(dev_matrix_payload, step=steps)
                            # Print headline cells for human visibility
                            for hk in ("dev/comm_mel/L1_clean", "dev/comm_mel/L3_clean"):
                                if hk in dev_matrix_payload:
                                    self.print(f"  {hk}: {dev_matrix_payload[hk]:0.4f}")

                        # ---- TB upgrade #6 + #12: dev-matrix audio + spectrogram for select combos ----
                        if dev_subset_xv and self._dev_audio_combos:
                            n_audio = min(self.showpiece_num, len(dev_subset_xv))
                            for L_eval, cond_name in self._dev_audio_combos:
                                # Find the cond dict by name
                                cond_eval = next(
                                    (c for c in self.channel_conditions if c["name"] == cond_name),
                                    None,
                                )
                                if cond_eval is None:
                                    continue
                                for i_a in range(n_audio):
                                    xv_dev = dev_subset_xv[i_a].to(self.device)
                                    full_codes_dev = gen_unwrapped_eval.encode(
                                        xv_dev, n_q=int(self.cfg["n_q"]), st=0
                                    )
                                    truncated_dev = full_codes_dev[:int(L_eval)].contiguous().long()
                                    pert_dev, _ = apply_channel_sim_torch(
                                        truncated_dev,
                                        codebook_size=self.codebook_size,
                                        p_drop=float(cond_eval.get("p_drop", 0.0)),
                                        p_sub=float(cond_eval.get("p_sub", 0.0)),
                                        generator=self._device_generator(),
                                    )
                                    y_comm = gen_unwrapped_eval.decode(pert_dev, st=0)
                                    tag_audio = (
                                        f"generate_comm/x_hat_comm_{i_a}_L{int(L_eval)}_{cond_name}"
                                    )
                                    tag_spec = (
                                        f"generate_comm/x_hat_comm_spec_{i_a}_L{int(L_eval)}_{cond_name}"
                                    )
                                    self.log(
                                        {tag_audio: y_comm[0].cpu().detach()},
                                        type="audio",
                                        sample_rate=self.sample_rate,
                                        step=steps,
                                    )
                                    yc_spec = mel_spectrogram(y_comm.squeeze(1), **self.mel_kwargs)
                                    self.log(
                                        {tag_spec: plot_spectrogram(yc_spec[0].cpu().numpy())},
                                        type="figure",
                                        step=steps,
                                    )

                    model_path = str(self.results_folder / f"SpeechTokenizerTrainer_{steps:08d}")
                    self.save(model_path, total_mel_error / max(1, num))
                    self.print(f"{steps}: saving model to {str(self.results_folder)}")
                    self.generator.train()

                # ============= Step + warmup =============
                self.steps += 1
                steps = int(self.steps.item())
                if steps < self.num_warmup_steps:
                    lr = self.warmup(steps)
                    for pg in self.optim_g.param_groups:
                        pg["lr"] = lr
                    for pg in self.optim_d.param_groups:
                        pg["lr"] = lr
                else:
                    lr = self.scheduler_g.get_last_lr()[0]

        self.print("training complete")
        self._close_lca_metrics()
