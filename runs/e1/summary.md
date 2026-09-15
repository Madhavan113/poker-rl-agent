# E1 Kuhn — summary

1024 sessions per agent = 256 opponents × 4 sessions, H = 64 hands per session, agents: Equilibrium, Thompson, BayesBR, Transformer(sample), Transformer(argmax), Random, OracleBR. All EV / exploitability numbers are exact (game-tree traversal of the agent's hand-t strategy); ± is one standard error across sessions. Regret is cumulative EV shortfall versus `OracleBR` on paired sessions (same opponent, same deals).

## Agents

| agent | sessions | EV first 8 hands | EV last 16 hands | EV all hands | cum. regret vs OracleBR (t=H) | mean exploitability | final KL (nats) |
|---|---|---|---|---|---|---|---|
| Equilibrium | 1024 | +0.0539 ± 0.0018 | +0.0539 ± 0.0018 | +0.0539 ± 0.0018 | +9.514 ± 0.212 | +0.0000 ± 0.0000 | – |
| Thompson | 1024 | +0.0864 ± 0.0032 | +0.1864 ± 0.0050 | +0.1634 ± 0.0045 | +2.506 ± 0.052 | +0.3148 ± 0.0022 | -0.000 ± 0.000 |
| BayesBR | 1024 | +0.1242 ± 0.0039 | +0.1930 ± 0.0050 | +0.1779 ± 0.0047 | +1.576 ± 0.049 | +0.3196 ± 0.0025 | -0.000 ± 0.000 |
| Transformer(sample) | 1024 | +0.0824 ± 0.0029 | +0.1677 ± 0.0045 | +0.1478 ± 0.0040 | +3.508 ± 0.104 | +0.2484 ± 0.0027 | +1.498 ± 0.030 |
| Transformer(argmax) | 1024 | +0.1123 ± 0.0034 | +0.1688 ± 0.0046 | +0.1573 ± 0.0042 | +2.899 ± 0.131 | +0.3242 ± 0.0027 | +1.742 ± 0.030 |
| Random | 1024 | -0.1346 ± 0.0033 | -0.1346 ± 0.0033 | -0.1346 ± 0.0033 | +21.578 ± 0.322 | +0.4583 ± 0.0000 | – |
| OracleBR | 1024 | +0.2026 ± 0.0048 | +0.2026 ± 0.0048 | +0.2026 ± 0.0048 | +0.000 ± 0.000 | +0.3148 ± 0.0030 | – |

## Success criteria (docs/experiments/e1-kuhn.md)

| check | criterion | verdict | numbers |
|---|---|---|---|
| H1_kl | KL(exact ‖ transformer belief) < 0.1 nats from hand 32 on | FAIL | scaled_to_H=False, kl_at_t=1.0756, kl_at_t_se=0.0258, kl_max_from_t=1.4977 |
| H1_entropy_tracking | belief entropy within 0.2 nats of the exact posterior entropy at every hand | FAIL | max_abs_gap=1.5902, t_of_max_gap=63, mean_abs_gap=0.9279 |
| H2_within_bayes | transformer EV within 0.02 chips/hand of BayesBR for t ≥ 16 (paired) | FAIL | scaled_to_H=False, n_paired=1024, max_shortfall=0.0413, mean_shortfall=0.0276, mean_shortfall_se=0.0003 |
| H2_beats_equilibrium | transformer EV above Equilibrium for every t ≥ 4 (paired) | PASS | scaled_to_H=False, n_paired=1024, min_advantage=0.0334, t_of_min=4, mean_advantage=0.0995, n_hands_failing=0 |
| sanity_oracle_dominates | OracleBR ≥ every agent at every hand of every paired session | PASS | tolerance=0.0000, max_violation=0.0000, worst_agent=– |
| sanity_equilibrium_exploitability | Equilibrium agent exploitability < 0.001 at every hand | PASS | expl_mean=0.0000, expl_max=0.0001 |
| sanity_equilibrium_vs_nash | Equilibrium agent EV vs an exact Nash opponent is ∓1/18 by seat | PASS | ev_seat0=-0.0556, ev_seat1=0.0556, target=[-0.05555555555555555, 0.05555555555555555], max_abs_error=0.0000 |

Resolved agent names: {'transformer': 'Transformer(sample)', 'bayes': 'BayesBR', 'equilibrium': 'Equilibrium', 'oracle': 'OracleBR'}. Hand thresholds are clipped to H − 1 when H is shorter than the spec's 64 hands (`scaled_to_H=True`).

## EV over all hands by opponent archetype

| agent | EV vs equilibrium | EV vs maniac | EV vs station | EV vs rock | EV vs random |
|---|---|---|---|---|---|
| Equilibrium | +0.0064 ± 0.0002 | +0.0856 ± 0.0009 | +0.0954 ± 0.0011 | +0.0099 ± 0.0004 | +0.1555 ± 0.0041 |
| Thompson | +0.0116 ± 0.0008 | +0.2195 ± 0.0023 | +0.2407 ± 0.0030 | +0.1515 ± 0.0026 | +0.4325 ± 0.0120 |
| BayesBR | +0.0177 ± 0.0011 | +0.2319 ± 0.0024 | +0.2620 ± 0.0025 | +0.1748 ± 0.0023 | +0.4501 ± 0.0127 |
| Transformer(sample) | +0.0090 ± 0.0010 | +0.2088 ± 0.0032 | +0.2160 ± 0.0045 | +0.1538 ± 0.0027 | +0.3571 ± 0.0122 |
| Transformer(argmax) | +0.0110 ± 0.0012 | +0.2160 ± 0.0043 | +0.2471 ± 0.0039 | +0.1706 ± 0.0025 | +0.3459 ± 0.0141 |
| Random | -0.1297 ± 0.0012 | -0.3293 ± 0.0017 | -0.1164 ± 0.0019 | -0.0712 ± 0.0014 | -0.0136 ± 0.0115 |
| OracleBR | +0.0453 ± 0.0009 | +0.2504 ± 0.0021 | +0.2792 ± 0.0023 | +0.1882 ± 0.0021 | +0.5041 ± 0.0127 |

## Training configuration (from the checkpoint)

| field | value (from the checkpoint) |
|---|---|
| checkpoint_step | 12000 |
| steps | 12000 |
| batch_size | 32 |
| lr | 0.000300 |
| warmup | 500 |
| weight_decay | 0.010000 |
| lambda_opp | 0.500000 |
| device | auto |
| seed | 0 |
| data_dir | data/e1a |
| model | {"d_model": 128, "n_layers": 4, "n_heads": 4, "max_len": 1024, "n_opp": 256} |
| data | {"n_sessions": 100000, "H": 64, "L": 640, "seed": 0, "collection_mix": {"equilibrium": 0.4, "oracle": 0.3, "random": 0.3}} |
| population_content_sha256 | eaaaad249bb36c40be6ce4a531ad57c409ff74ae2bd5b3e4b492e52cc038434b |

## Notes

- "final KL" is KL(exact ‖ belief) *before* the last hand, i.e. after H − 1 = 63 observed hands (the belief after the final hand is not recorded).
- KL floors the agent's belief at 1e-30 wherever the exact posterior has mass, so a zero-mass miss contributes at most −log(1e-30) ≈ 69.08 nats per element and is never infinite.
- Criteria are evaluated pointwise on per-hand means for every t ≥ threshold (stricter than the spec's "by t = 32 on average"); hand thresholds are clipped to H − 1 for short runs (`scaled_to_H=True`).
- Regret pairs sessions by (opponent, session index): every agent faced the same deals.
- Training defaults are batch 32 x 12000 steps instead of the spec's 64 x 20000: measured MPS throughput for the 4 x 128 model at L = 640 is ~100-135 session-passes/s, so the spec budget would take several hours; --batch/--steps override the defaults.

## Figures

`ev_vs_hand.png`, `entropy_vs_hand.png`, `exploitability_vs_hand.png`, `regret_cumulative.png`, `probes.png` (all regenerable from `summary.json`).

## Run

- **command**: uv run python -m exsolver.experiments.e1_kuhn eval --out runs/e1 --data data/e1a --device auto --no-progress
- **config**: {"out": "runs/e1", "data": "data/e1a", "m": 256, "hands": 64, "sessions": 100000, "steps": 12000, "batch": 32, "eval_sessions_per_opp": 4, "seed": 0, "eval_seed": 1, "device": "auto", "smoke": false, "max_train_minutes": null, "workers": 8, "retrain": false, "cfr_iterations": 2000, "d_model": 128, "n_layers": 4, "n_heads": 4, "progress": false, "extra_train": {}, "data_dir": "data/e1a"}
- **checkpoint**: runs/e1/model/model.pt
- **checkpoint_step**: 12000
- **population_content_sha256**: eaaaad249bb36c40be6ce4a531ad57c409ff74ae2bd5b3e4b492e52cc038434b
- **device**: mps
- **eval_seconds**: 1621.8
- **sessions_file**: runs/e1/sessions.npz
