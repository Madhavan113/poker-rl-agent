# E5 — population node-locking against static and adaptive opponents

Status: spec (2026-11-26). Phase 1 safety experiment (RESEARCH.md §2.6, §5 E5), brought forward and
reframed around a practitioner's question.

## Question

A "population solver": infer the opponent's (unbalanced) strategy from the hands seen so far, lock
the solver's tree to it, and re-solve with CFR. How much does that earn over plain CFR (Nash), how
does it compare with simply best-responding to the inferred opponent, and does any of it survive an
opponent who adapts to us?

Locking and plain CFR use the same algorithm; they differ in the game being solved. Plain CFR solves
the full game (opponent free everywhere, result: Nash). Node-locking solves a *restricted* game in
which the opponent is fixed to the model at some infosets and free to counter-adjust at the rest.
That freedom is the point (robustness) and the catch: an "unbalanced range" at a node is the product
of the opponent's earlier decisions, so a lock on one node is undone if the earlier nodes are left
free and the solver lets the opponent re-route hands around it.

## Prototype evidence (Kuhn, scratch script, 2026-11-26)

A batched constrained CFR+ on Kuhn, 256 E1a opponents, `n` hands observed while playing Nash, then a
fixed strategy; EV exact against the true opponent. Gain over Nash (chips/hand) / exploitability:

| n | full lock (Bayes) | full lock (EM) | hard lock n_I ≥ 5 (EM) | RNR p = .8 (Bayes) | RNR p = .5 (Bayes) |
|---|---|---|---|---|---|
| 16 | +0.124 / 0.33 | +0.067 / 0.32 | +0.001 / 0.00 | +0.104 / 0.13 | +0.060 / 0.03 |
| 64 | +0.144 / 0.32 | +0.115 / 0.30 | +0.081 / 0.20 | +0.127 / 0.15 | +0.069 / 0.03 |
| 1024 | +0.151 / 0.32 | +0.145 / 0.32 | +0.145 / 0.31 | +0.134 / 0.17 | +0.069 / 0.03 |

(Nash +0.052; oracle best response +0.151 over Nash.) Readings: the population prior roughly doubles
early EV over a prior-free fit; partial node locks are near-worthless at small samples; RNR mixing is
the better robustness knob. Against static opponents the full lock is simply best. This experiment
asks whether the ranking holds in Leduc and whether it flips against adaptive opponents. The script
is not part of the repository, so these numbers are indicative only and are not carried into the
results; the E5 runner supersedes it.

## Restricted games (implemented in `exsolver.solvers.constrained`)

For our seat `s` and an opponent model `θ̂` (a behavioural strategy for the other seat):

- **Node lock with confidence `c`**, one weight per opponent infoset: the opponent plays
  `c_I θ̂(I) + (1 − c_I) τ(I)` and chooses `τ` adversarially. `c ≡ 1` is the best response to `θ̂`
  (`br`), `c ≡ 0` is Nash, `c ∈ {0,1}` is a solver-style **hard** lock (`c_I = 1[n_I ≥ m]`), and
  `c_I = n_I / (n_I + k)` is the data-biased-response style **soft** lock (Johanson & Bowling 2009).
  `n_I` is the (weighted) expected number of opponent decisions observed at `I` under `θ̂`.
- **Restricted Nash response** `RNR(p)` (Johanson, Zinkevich & Bowling 2007): maximise
  `p · EV(σ, θ̂) + (1 − p) · min_τ EV(σ, τ)`; the model and the adversary are mixed per hand.

Both are zero-sum games over convex strategy sets. Solver: batched CFR+ (alternating updates,
regret matching+ on `τ` and on our strategy, linear averaging of ours), warm-started across
re-solves (regrets persist, the average restarts). `restricted_value` and `rnr_objective` evaluate
the two objectives exactly for any strategy; `duality_gap` bounds a solution's distance from the
restricted game's value using the adversary's average strategy (lower bound: our objective; upper
bound: our best response to the adversary).

## Population (Leduc, `exsolver.population.leduc_prior`)

Perturbed CFR (RESEARCH.md §3.3). From the CFR+ equilibrium `s*` (2000 iterations), smoothed
`s~ = 0.95 s* + 0.05 uniform`:

    logit(a | I) = log s~(a | I) + β · b_arch(a, situation(I)) + ε(I, a),   ε ~ N(0, 0.15²),  β ~ U[0.5, 1.5]

`situation` = (facing a bet?, hand-strength bucket); buckets: preflop J/Q/K = weak/mid/strong,
postflop pair = strong, unpaired K = mid, else weak. Five archetypes at weight 0.2 each. Their logit
biases `b_arch` on (fold, call, raise), exactly as in `default_leduc_archetypes` (amended 2026-11-26
to match the code the E5 run used; the first version of this section paraphrased it loosely; FOLD is
illegal when not facing a bet):

| archetype | not facing a bet: weak / mid / strong | facing a bet: weak / mid / strong |
|---|---|---|
| equilibrium | 0 everywhere | 0 everywhere |
| maniac | raise +1.5 / +1.5 / +1.0 | fold −1.0, raise +1.0 (all buckets) |
| station | call +0.5 (all buckets) | (−2.0, +1.5, −0.5) / (−2.0, +1.5, −0.5) / (−1.0, +1.0, −0.5) |
| rock | raise −1.5 / −0.5 / 0 | (+1.5, 0, −1.5) / (+1.0, 0, −0.5) / 0 |
| passive | raise −2.0 / −2.0 / −1.5 | call +0.5 and raise −2.0 / −2.0 / −1.5 |

Population `M = 256` (seed `[seed, 1]`). Opponent sets:
**in-pop**, 64 evenly spaced members of the population (the Bayes prior is well specified); **fresh**,
64 new draws from the same prior (seed `[seed, 2]`; the prior's support misses them).

Kuhn runs use the E1 Kuhn prior through the same code path (`--game kuhn`).

## Adaptive opponents (`exsolver.population.adaptive`)

After every block `k` the opponent moves toward a best response to the strategy `σ_k` we played:

    θ_{k+1} = (1 − η) θ_k ⊕ η BR(σ_k)          (⊕ = sequence-form mixture)

`η ∈ {0, 0.1, 0.3}`. The adapter sees `σ_k` exactly (omniscient, the fastest learner of this family
for a given `η`); it is the worst case for an exploiter and the best case for Nash, whose BR earns
the adapter nothing. A realistic adapter that must estimate `σ_k` from observations is out of scope.

## Agent protocol

Sessions of 16 blocks × 32 hands (512 hands), alternating seats. Block 0 plays the equilibrium.
Before every later block the agent estimates `θ̂` from all hands so far with per-hand weight
`γ^{age in blocks}`, `γ ∈ {1, 0.8}`, builds its strategy, and plays the block. Inference
(`exsolver.bayes.online`):

- **bayes**: discounted posterior over the 256-member population (prior uniform), estimate = its
  sequence-form mixture. Under adaptation or for fresh opponents the model is misspecified.
- **em**: prior-free posterior mode with 4 · `s~` pseudo-observations per opponent infoset (the
  MAP under Dirichlet(4 · `s~` + 1)), EM over the hidden card,
  30 iterations warm-started from the previous block's estimate.

Methods: `nash`, `br`, `hard` (m = 10), `soft` (k = 10), `rnr_lo` (p = 0.5), `rnr_hi` (p = 0.8).
Re-solves: 300 CFR+ iterations, cold (a warm start after a large lock change can be slightly
worse than a cold one). Every re-solve's certified duality gap (`duality_gap`) is recorded and its
mean and maximum are reported, so solver error is bounded for every number.

## Metrics (all exact)

Per block: EV of our block strategy against the opponent's *current* profile, averaged over the two
seats; exploitability of our block strategy (mean over seats of the worst-case loss below the game
value). Per config: mean EV over the session and over its second half, gain over `nash` paired by
opponent, mean exploitability, per-archetype gain. Static oracle: best response to each opponent's
base profile (`η = 0` reference). Uncertainty: ±1 s.e. over the 64 opponents. Deals and action
draws are seeded per (opponent, block, hand), identical across configs.

## Hypotheses

- **E5-1 (static ranking)**: at `η = 0`, in-pop, `br` ≥ `rnr_hi` ≥ `soft` ≥ `hard` ≥ `nash` in
  session EV, as in the Kuhn prototype.
- **E5-2 (adaptation flips it)**: at `η = 0.3`, `br`'s second-half gain over `nash` is negative,
  and the best second-half gain belongs to an RNR config.
- **E5-3 (forgetting)**: under adaptation `γ = 0.8` improves `br`'s second-half EV more than it
  improves `rnr_*`'s.
- **E5-4 (prior)**: in-pop, `bayes` beats `em` for every method over the session; on fresh
  opponents the gap shrinks.
- **E5-5 (partial locks)**: in Leduc, with 468 opponent infosets per seat (erratum: first written
  as ~234; infosets keep suits, so counts split across suit-equivalent infosets) and 512 hands, hard and
  soft locks stay within 0.05 chips/hand of `nash`; the robustness they buy is bought more cheaply
  by RNR.

## Outputs and budget

`uv run python -m exsolver.experiments.e5_nodelock --game leduc --out runs/e5` writes
`summary.json`, `summary.md`, `blocks.npz` (per-opponent block EVs and exploitabilities), the
EV-by-block figures and the gain-vs-exploitability frontier, and appends the command line and
config to `runs/e5/README.md`. 126 configs × 64 opponents; on the M4 with 8 workers, about an hour.
Results go in `e5-population-nodelock-results.md` in a separate PR.

## Out of scope

Opponents that model us from observations; information-seeking play; the transformer agent
(E1/E2) as the inference engine, which is the natural follow-up once the exact references exist.
