
## run — 2026-11-27T01:50:09+00:00

```
uv run python -m exsolver.experiments.e5_nodelock --game leduc --out /Users/madhavanp/exploitative-solver/runs/e5 --workers 8
```

```json
{
 "game": "leduc",
 "out": "/Users/madhavanp/exploitative-solver/runs/e5",
 "m": 256,
 "n_opp": 64,
 "sets": [
  "in-pop",
  "fresh"
 ],
 "block": 32,
 "n_blocks": 16,
 "etas": [
  0.0,
  0.1,
  0.3
 ],
 "inferences": [
  "bayes",
  "em"
 ],
 "gammas": [
  1.0,
  0.8
 ],
 "methods": [
  "nash",
  "br",
  "hard",
  "soft",
  "rnr_lo",
  "rnr_hi"
 ],
 "hard_m": 10.0,
 "soft_k": 10.0,
 "rnr_lo": 0.5,
 "rnr_hi": 0.8,
 "em_strength": 4.0,
 "em_iterations": 30,
 "ref_smooth": 0.05,
 "cfr_iterations": 300,
 "eq_iterations": 2000,
 "seed": 0,
 "workers": 8,
 "smoke": false,
 "extra": {}
}
```

wall time 5960 s, git d4e34dc11a88f34ea65dc7126ceceb8153aaf9b6, figures: ev_by_block_in-pop_bayes.png, ev_by_block_in-pop_em.png, ev_by_block_fresh_bayes.png, ev_by_block_fresh_em.png, frontier.png
