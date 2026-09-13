"""DPT training loop: AdamW, linear warm-up + cosine decay, grad clipping, JSONL logs, checkpoints.

CLI: ``uv run python -m exsolver.train --data DIR --out DIR --steps N ...``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from exsolver.data.shards import ShardDataset, tokenizer_spec_from_meta
from exsolver.data.tokenizer import Tokenizer, TokenizerSpec
from exsolver.model.losses import compute_loss, to_floats
from exsolver.model.transformer import ExploitTransformer, ModelConfig

_HEAD_CHOICES = ("auto", "on", "off")

__all__ = [
    "TrainConfig",
    "build_model_config",
    "evaluate",
    "load_checkpoint",
    "lr_at",
    "main",
    "resolve_device",
    "tokenizer_from_checkpoint",
    "tokenizer_spec_from_meta",
    "train",
    "trim_padding",
]


@dataclass
class TrainConfig:
    """Everything needed to reproduce a run (stored in the checkpoint).

    Spec defaults (docs/experiments/e1-kuhn.md §Model): AdamW, lr 3e-4, weight decay 0.01,
    cosine decay, 500 warm-up steps, batch 64, 20k steps, device mps-else-cpu.  All schedule
    quantities (``steps``, ``warmup``, ``eval_every``, ``log_every``) are optimizer steps and are
    independent of ``batch_size``; nothing here assumes batch 64.

    Deviations from docs/experiments/e1-kuhn.md §Model (kept deliberately):

    - ``grad_clip=1.0``: global-norm gradient clipping; the spec is silent, and clipping is the
      standard guard against loss spikes during warm-up for small transformers.
    - ``lr_min_ratio=0.1``: the cosine decays to 10% of ``lr`` instead of 0 so the final steps
      still make progress; the spec only says "cosine decay".
    - ``betas=(0.9, 0.95)``: GPT-style second-moment decay (torch default 0.999) which is better
      behaved for small, noisy batches; the spec is silent.
    - weight decay only on >=2-D tensors (matrices, embeddings), none on biases / LayerNorm:
      standard practice; the spec gives "weight decay 0.01" without a scope.
    - ``trim_padding=True``: each batch is sliced to its longest real session; loss-exact under
      causal attention and ~1.7x faster on MPS where attention backward dominates.
    - held-out split = the last ``eval_size`` sessions (<= 10% of the data); the spec has none.

    Reproducibility: runs are bit-reproducible on CPU for a given ``seed``; MPS kernels are not
    bit-deterministic, so MPS runs reproduce only statistically.
    """

    data_dir: str
    out_dir: str
    steps: int = 20_000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 500
    lambda_opp: float = 0.5
    lambda_theta: float = 0.5
    # model
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_len: int = 1024
    dropout: float = 0.0
    opp_head: str = "auto"  # auto: on iff every session has opp_id >= 0 (discrete population)
    theta_head: str = "auto"  # auto: on iff the opponent head is off and theta has P > 0
    # run
    seed: int = 0
    device: str = "auto"  # auto: mps if available else cpu
    eval_every: int = 500
    log_every: int = 50
    eval_size: int = 512  # held-out sessions (capped at 10% of the data)
    grad_clip: float = 1.0
    lr_min_ratio: float = 0.1  # cosine decays to lr * lr_min_ratio
    betas: tuple[float, float] = (0.9, 0.95)
    trim_padding: bool = True  # slice each batch to its longest real session (loss-exact)
    progress: bool = True

    def __post_init__(self) -> None:
        if self.opp_head not in _HEAD_CHOICES or self.theta_head not in _HEAD_CHOICES:
            raise ValueError(f"opp_head/theta_head must be one of {_HEAD_CHOICES}")
        self.betas = tuple(self.betas)  # type: ignore[assignment]


# ---------------------------------------------------------------------- small helpers
def resolve_device(name: str = "auto") -> torch.device:
    """``mps`` if available else ``cpu`` for ``"auto"``; otherwise the named device."""
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warm-up to ``lr`` then cosine decay to ``lr * lr_min_ratio`` at ``steps``."""
    if cfg.warmup > 0 and step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    decay_steps = max(cfg.steps - cfg.warmup, 1)
    progress = min(max(step - cfg.warmup, 0) / decay_steps, 1.0)
    floor = cfg.lr * cfg.lr_min_ratio
    return floor + 0.5 * (cfg.lr - floor) * (1.0 + math.cos(math.pi * progress))


def build_model_config(cfg: TrainConfig, ds: ShardDataset) -> ModelConfig:
    """Derive vocab size and head sizes from ``meta.json`` (falling back to shard 0 and the
    per-shard ``opp_id`` arrays, which are read without loading full shards) and the head switches."""
    meta = ds.meta
    L = ds.L
    if L > cfg.max_len:
        raise ValueError(f"data L={L} exceeds max_len={cfg.max_len}")
    tspec = tokenizer_spec_from_meta(meta)
    if "vocab_size" in meta:
        vocab_size = int(meta["vocab_size"])
    elif tspec is not None:
        vocab_size = tspec.vocab_size
    else:
        vocab_size = int(ds._store.shard(0).tokens.max()) + 1
    opp_id = ds.opp_ids
    discrete = bool(opp_id.size) and bool(opp_id.min() >= 0)
    use_opp = cfg.opp_head == "on" or (cfg.opp_head == "auto" and discrete)
    n_opp: int | None = None
    if use_opp:
        n_opp = int(meta.get("n_opp") or meta.get("population_size") or (int(opp_id.max()) + 1))
        if n_opp < 1:
            raise ValueError("opponent head requested but no opponent ids in the data")
    P = ds.P
    use_theta = cfg.theta_head == "on" or (cfg.theta_head == "auto" and not use_opp and P > 0)
    if use_theta and P < 1:
        raise ValueError("theta head requested but theta has no parameters")
    return ModelConfig(
        vocab_size=vocab_size,
        n_opp=n_opp,
        theta_dim=P if use_theta else None,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
    )


def make_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices/embeddings only (not biases or LayerNorm)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


_PER_POSITION = ("tokens", "action_mask", "action_target", "legal", "belief_mask")


def trim_padding(
    batch: dict[str, torch.Tensor], pad_id: int = 0, multiple: int = 8
) -> dict[str, torch.Tensor]:
    """Cut the sequence axis to the longest non-PAD session in the batch (rounded up to ``multiple``).

    Exact for the loss: trailing PAD positions carry no mask, and causal attention means earlier
    positions never see them.  Attention cost is quadratic in length, so this is a large saving
    when sessions are much shorter than ``L``.
    """
    tokens = batch["tokens"]
    full_len = tokens.shape[1]
    real = tokens != pad_id
    # Last non-PAD position per row (argmax of the flipped row finds the first True from the
    # right); rows without any real token conservatively keep the full length.
    last = full_len - 1 - real.flip(1).to(torch.int8).argmax(dim=1)
    last = torch.where(real.any(dim=1), last, torch.full_like(last, full_len - 1))
    t_eff = int(last.max().item()) + 1
    t_eff = min(full_len, -(-t_eff // multiple) * multiple)
    if t_eff >= full_len:
        return batch
    return {k: (v[:, :t_eff] if k in _PER_POSITION else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    model: ExploitTransformer,
    ds: ShardDataset,
    device: torch.device,
    batch_size: int = 128,
    lambda_opp: float = 0.5,
    lambda_theta: float = 0.5,
    trim: bool = True,
) -> dict[str, float]:
    """Loss components over ``ds``: action metrics are per-decision means, opponent / theta
    metrics per-belief-position means, the total loss a per-session mean; counts are summed.
    NaN batch accuracies (no masked positions) are skipped."""
    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    weights: dict[str, float] = {}
    for start in range(0, len(ds), batch_size):
        idx = np.arange(start, min(start + batch_size, len(ds)))
        batch = ds.get_batch(idx)
        if trim:
            batch = trim_padding(batch)
        batch = to_device(batch, device)
        _, comps = compute_loss(model(batch["tokens"]), batch, lambda_opp, lambda_theta)
        vals = to_floats(comps)
        for k, v in vals.items():
            if k.startswith("n_"):
                sums[k] = sums.get(k, 0.0) + v
                continue
            w = vals.get(_WEIGHT_KEY[k], 0.0) if k in _WEIGHT_KEY else float(idx.size)
            if w <= 0 or math.isnan(v):
                continue
            sums[k] = sums.get(k, 0.0) + v * w
            weights[k] = weights.get(k, 0.0) + w
    if was_training:
        model.train()
    return {
        k: (v if k.startswith("n_") else v / weights[k])
        for k, v in sums.items()
        if k.startswith("n_") or k in weights
    }


_WEIGHT_KEY = {
    "loss_action": "n_action",
    "acc_action": "n_action",
    "loss_opp": "n_belief",
    "acc_opp": "n_belief",
    "loss_theta": "n_belief",
}


# ---------------------------------------------------------------------- checkpoints
def save_checkpoint(
    path: str | Path,
    model: ExploitTransformer,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    tokenizer_spec: TokenizerSpec | None,
    data_meta: dict[str, Any],
    step: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``model.pt`` with config + tokenizer spec + data meta + weights (weights_only-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_config": model_cfg.to_dict(),
        "train_config": _plain(asdict(train_cfg)),
        "tokenizer_spec": None if tokenizer_spec is None else asdict(tokenizer_spec),
        "data_meta": _plain(data_meta),
        "step": int(step),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if extra:
        payload.update(_plain(extra))
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[ExploitTransformer, ModelConfig, dict[str, Any]]:
    """Rebuild the model from ``model.pt``; returns ``(model, model_config, meta)``.

    ``meta`` holds ``tokenizer_spec``, ``train_config``, ``data_meta`` and ``step``.
    """
    device = torch.device(device)
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model_cfg = ModelConfig.from_dict(payload["model_config"])
    model = ExploitTransformer(model_cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, model_cfg, meta


def tokenizer_from_checkpoint(meta: dict[str, Any]) -> Tokenizer:
    """Tokenizer matching a checkpoint's stored ``tokenizer_spec``."""
    spec = meta.get("tokenizer_spec")
    if not spec:
        raise ValueError("checkpoint has no tokenizer_spec")
    return Tokenizer(TokenizerSpec(int(spec["n_cards"]), int(spec["max_result"])))


def _plain(obj: Any) -> Any:
    """Recursively convert to JSON/``weights_only``-safe builtins."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.integer | np.floating | np.bool_):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------------- training
class JsonlLogger:
    """One JSON object per line in ``path`` (truncated at the start of each run)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")

    def log(self, record: dict[str, Any]) -> None:
        self._f.write(json.dumps(_plain(record)) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def train(cfg: TrainConfig) -> dict[str, Any]:
    """Run the full training loop; returns a summary with final eval metrics and paths."""
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ShardDataset(cfg.data_dir)
    n_eval = min(cfg.eval_size, len(ds) // 10)
    train_ds, eval_ds = ds.split(n_eval)
    if len(train_ds) == 0:
        raise ValueError("no training sessions")
    model_cfg = build_model_config(cfg, ds)
    tspec = tokenizer_spec_from_meta(ds.meta)
    model = ExploitTransformer(model_cfg).to(device)
    opt = make_optimizer(model, cfg)
    logger = JsonlLogger(out_dir / "log.jsonl")
    ckpt_path = out_dir / "model.pt"
    logger.log(
        {
            "event": "start",
            "train_config": asdict(cfg),
            "model_config": model_cfg.to_dict(),
            "n_params": model.num_parameters(),
            "n_train": len(train_ds),
            "n_eval": len(eval_ds),
            "device": str(device),
            # CPU runs are bit-reproducible for a seed; MPS kernels are not bit-deterministic.
            "bit_reproducible": device.type == "cpu",
        }
    )

    def run_eval(step: int) -> dict[str, float]:
        if len(eval_ds) == 0:
            return {}
        metrics = evaluate(
            model,
            eval_ds,
            device,
            2 * cfg.batch_size,
            cfg.lambda_opp,
            cfg.lambda_theta,
            cfg.trim_padding,
        )
        logger.log({"step": step, "split": "eval", **metrics, "elapsed": time.time() - t0})
        return metrics

    def save(step: int) -> None:
        save_checkpoint(ckpt_path, model, model_cfg, cfg, tspec, ds.meta, step)

    perm = rng.permutation(len(train_ds))
    cursor = 0
    t0 = time.time()
    t_log = t0
    eval_metrics: dict[str, float] = {}
    model.train()
    bar = tqdm(range(cfg.steps), disable=not cfg.progress, dynamic_ncols=True, file=sys.stderr)
    for step in bar:
        if cursor + cfg.batch_size > perm.size:
            perm = rng.permutation(len(train_ds))
            cursor = 0
        idx = perm[cursor : cursor + cfg.batch_size]
        cursor += cfg.batch_size
        batch = train_ds.get_batch(idx)
        if cfg.trim_padding:
            batch = trim_padding(batch)
        batch = to_device(batch, device)

        lr = lr_at(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        loss, comps = compute_loss(model(batch["tokens"]), batch, cfg.lambda_opp, cfg.lambda_theta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        is_last = step + 1 == cfg.steps
        if (step + 1) % cfg.log_every == 0 or is_last:
            now = time.time()
            vals = to_floats(comps)
            record = {
                "step": step + 1,
                "split": "train",
                "lr": lr,
                **vals,
                "grad_norm": float(grad_norm),
                "elapsed": now - t0,
                "steps_per_sec": cfg.log_every / max(now - t_log, 1e-9) if not is_last else None,
            }
            logger.log(record)
            bar.set_postfix(loss=f"{vals['loss']:.3f}", acc=f"{vals['acc_action']:.3f}")
            t_log = now
        if cfg.eval_every > 0 and ((step + 1) % cfg.eval_every == 0 or is_last):
            eval_metrics = run_eval(step + 1)
            save(step + 1)
    if not (cfg.eval_every > 0):
        eval_metrics = run_eval(cfg.steps)
        save(cfg.steps)
    elapsed = time.time() - t0
    summary = {
        "steps": cfg.steps,
        "elapsed": elapsed,
        "steps_per_sec": cfg.steps / max(elapsed, 1e-9),
        "n_params": model.num_parameters(),
        "eval": eval_metrics,
        "checkpoint": str(ckpt_path),
        "log": str(logger.path),
        "model_config": model_cfg.to_dict(),
    }
    logger.log({"event": "end", **summary})
    logger.close()
    return summary


# ---------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the exploitative transformer on shards.")
    p.add_argument("--data", required=True, help="directory of .npz shards (+ meta.json)")
    p.add_argument("--out", required=True, help="output directory (log.jsonl, model.pt)")
    skip = {"data_dir", "out_dir", "betas", "opp_head", "theta_head", "progress", "trim_padding"}
    for f in fields(TrainConfig):
        if f.name in skip:
            continue
        flag = "--" + f.name.replace("_", "-")
        p.add_argument(flag, type=type(f.default), default=f.default, help=f"default: {f.default}")
    p.add_argument("--opp-head", choices=_HEAD_CHOICES, default="auto")
    p.add_argument("--theta-head", choices=_HEAD_CHOICES, default="auto")
    p.add_argument(
        "--betas", type=float, nargs=2, default=list(TrainConfig.betas), metavar=("B1", "B2")
    )
    p.add_argument("--no-trim-padding", action="store_true", help="always run on the full L")
    p.add_argument("--no-progress", action="store_true", help="disable the tqdm bar")
    return p


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    kwargs = {
        k: v
        for k, v in vars(args).items()
        if k not in {"data", "out", "no_progress", "no_trim_padding"}
    }
    kwargs["betas"] = tuple(kwargs["betas"])
    return TrainConfig(
        data_dir=args.data,
        out_dir=args.out,
        progress=not args.no_progress,
        trim_padding=not args.no_trim_padding,
        **kwargs,
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    cfg = config_from_args(build_parser().parse_args(argv))
    summary = train(cfg)
    print(json.dumps({k: v for k, v in summary.items() if k != "model_config"}, indent=2))
    return summary


if __name__ == "__main__":
    main()
