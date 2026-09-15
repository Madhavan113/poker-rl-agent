# runs/e1

Every stage appends its exact command line here so each number in `summary.md` can be regenerated.

## train — 2026-09-13T05:35:44

```
uv run python -m exsolver.experiments.e1_kuhn train --out runs/e1 --data data/e1a --steps 12000 --batch 32 --device auto --no-progress
```

```json
{
  "config": {
    "out": "runs/e1",
    "data": "data/e1a",
    "m": 256,
    "hands": 64,
    "sessions": 100000,
    "steps": 12000,
    "batch": 32,
    "eval_sessions_per_opp": 4,
    "seed": 0,
    "eval_seed": 1,
    "device": "auto",
    "smoke": false,
    "max_train_minutes": null,
    "workers": 8,
    "cfr_iterations": 2000,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "progress": false,
    "extra_train": {},
    "data_dir": "data/e1a"
  },
  "train_config": {
    "data_dir": "data/e1a",
    "out_dir": "runs/e1/model",
    "steps": 12000,
    "batch_size": 32,
    "lr": 0.0003,
    "weight_decay": 0.01,
    "warmup": 500,
    "lambda_opp": 0.5,
    "lambda_theta": 0.5,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "max_len": 1024,
    "dropout": 0.0,
    "opp_head": "auto",
    "theta_head": "auto",
    "seed": 0,
    "device": "auto",
    "eval_every": 500,
    "log_every": 50,
    "eval_size": 512,
    "grad_clip": 1.0,
    "lr_min_ratio": 0.1,
    "betas": [
      0.9,
      0.95
    ],
    "trim_padding": true,
    "progress": false
  },
  "summary": {
    "steps": 12000,
    "elapsed": 4077.6919753551483,
    "steps_per_sec": 2.942841213246583,
    "n_params": 961283,
    "eval": {
      "loss_action": 0.24015439633051747,
      "acc_action": 0.8809454720578871,
      "n_action": 37907.0,
      "n_belief": 33280.0,
      "loss_opp": 3.660523474216461,
      "acc_opp": 0.1315504815429449,
      "loss": 2.070332944393158
    },
    "checkpoint": "runs/e1/model/model.pt",
    "log": "runs/e1/model/log.jsonl",
    "budget": {
      "max_train_minutes": null,
      "requested_steps": 12000
    }
  },
  "elapsed_s": 4078.4197297499923
}
```

## eval — 2026-09-13T06:03:00

```
uv run python -m exsolver.experiments.e1_kuhn eval --out runs/e1 --data data/e1a --device auto --no-progress
```

```json
{
  "config": {
    "out": "runs/e1",
    "data": "data/e1a",
    "m": 256,
    "hands": 64,
    "sessions": 100000,
    "steps": 12000,
    "batch": 32,
    "eval_sessions_per_opp": 4,
    "seed": 0,
    "eval_seed": 1,
    "device": "auto",
    "smoke": false,
    "max_train_minutes": null,
    "workers": 8,
    "retrain": false,
    "cfr_iterations": 2000,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "progress": false,
    "extra_train": {},
    "data_dir": "data/e1a"
  },
  "skipped": false,
  "n_sessions": 7168,
  "eval_seconds": 1621.800563375,
  "criteria": {
    "H1_kl": false,
    "H1_entropy_tracking": false,
    "H2_within_bayes": false,
    "H2_beats_equilibrium": true,
    "sanity_oracle_dominates": true,
    "sanity_equilibrium_exploitability": true,
    "sanity_equilibrium_vs_nash": true
  },
  "training": {
    "checkpoint_step": 12000,
    "steps": 12000,
    "batch_size": 32,
    "lr": 0.0003,
    "warmup": 500,
    "weight_decay": 0.01,
    "lambda_opp": 0.5,
    "device": "auto",
    "seed": 0,
    "data_dir": "data/e1a",
    "model": {
      "d_model": 128,
      "n_layers": 4,
      "n_heads": 4,
      "max_len": 1024,
      "n_opp": 256
    },
    "data": {
      "n_sessions": 100000,
      "H": 64,
      "L": 640,
      "seed": 0,
      "collection_mix": {
        "equilibrium": 0.4,
        "oracle": 0.3,
        "random": 0.3
      }
    },
    "population_content_sha256": "eaaaad249bb36c40be6ce4a531ad57c409ff74ae2bd5b3e4b492e52cc038434b",
    "spec_deviation": "Training defaults are batch 32 x 12000 steps instead of the spec's 64 x 20000: measured MPS throughput for the 4 x 128 model at L = 640 is ~100-135 session-passes/s, so the spec budget would take several hours; --batch/--steps override the defaults."
  },
  "outputs": [
    "runs/e1/ev_vs_hand.png",
    "runs/e1/entropy_vs_hand.png",
    "runs/e1/exploitability_vs_hand.png",
    "runs/e1/regret_cumulative.png",
    "runs/e1/probes.png",
    "runs/e1/summary.md",
    "runs/e1/summary.json"
  ],
  "elapsed_s": 1624.0833824590081
}
```
