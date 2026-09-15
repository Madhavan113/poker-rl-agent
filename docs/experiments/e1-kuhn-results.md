# E1 results — in-context Bayesian exploitation in Kuhn poker

Run of 2026-09-13 (`runs/e1/`; regenerate with the commands in `runs/e1/README.md`). Everything below is
exact (game-tree traversal), ± is one standard error over 1024 sessions per agent (256 opponents × 4
sessions, 64 hands, identical deals across agents). Model: 4 × 128 transformer, 0.96M parameters, trained
12 000 steps × batch 32 on 100 000 DPT sessions (68 min on the M4's MPS). Held-out action accuracy 0.881,
opponent-id accuracy 0.132 (chance 0.004), opponent-id loss still falling at the end of training.

## Headline numbers

| agent | EV all hands | EV last 16 hands | share of exploitable value, all / last 16 | cum. regret vs oracle | mean exploitability |
|---|---|---|---|---|---|
| Equilibrium (CFR+) | +0.054 | +0.054 | 0 / 0 | 9.51 | 0.000 |
| Thompson (exact posterior) | +0.163 | +0.186 | 0.74 / 0.89 | 2.51 | 0.315 |
| BayesBR (exact posterior, myopic Bayes) | +0.178 | +0.193 | 0.83 / 0.94 | 1.58 | 0.320 |
| Transformer (sample) | +0.148 | +0.168 | 0.63 / 0.77 | 3.51 | 0.248 |
| Transformer (argmax) | +0.157 | +0.169 | 0.70 / 0.77 | 2.90 | 0.324 |
| OracleBR (knows θ) | +0.203 | +0.203 | 1 / 1 | 0 | 0.315 |

"Share of exploitable value" = (EV − EV_equilibrium) / (EV_oracle − EV_equilibrium). Standard errors are
0.002–0.005 chips/hand on every EV entry (`runs/e1/summary.md`).

## What the criteria said

| criterion | verdict | reading |
|---|---|---|
| H2 beats equilibrium for every t ≥ 4 (paired) | **PASS** | min advantage +0.033 at t = 4, mean +0.100 |
| H2 within 0.02 of BayesBR for t ≥ 16 (paired) | FAIL | shortfall 0.028 mean, 0.041 max; within 0.02 of *Thompson* would pass for argmax |
| H1 KL(exact ‖ opponent head) < 0.1 nats from t = 32 | FAIL | KL grows to 1.5 nats; read-out entropy plateaus at 3.9 nats while the exact posterior falls 5.55 → 2.3 |
| H1 entropy tracking within 0.2 nats | FAIL | gap 0.93 mean, 1.59 at t = 63 |
| sanity: oracle dominates every session-hand; equilibrium exploitability ≈ 0; ∓1/18 vs Nash | **PASS** | exact |

## Findings

1. **In-context exploitation works and is worth a lot.** After 16 hands of context the transformer earns
   three times the equilibrium's edge (+0.17 vs +0.054 chips/hand) against opponents it has never
   seen identified, and it never falls below equilibrium after hand 4. Against the "rock" archetype it
   captures 0.15 of the 0.18 available, against maniacs 0.21 of 0.25.
2. **The action head is close to posterior sampling, not to the myopic Bayes-optimal policy.** Transformer
   (sample) sits just under Thompson and Transformer (argmax) just under Thompson's level too, while BayesBR
   is clearly better than both (+0.02–0.03 chips/hand). This is exactly the DPT label semantics (RESEARCH.md
   §2.5): the model learns the posterior over *best-response actions*, so sampling from it is Thompson
   sampling and argmax is the posterior-plurality action. The gap to BayesBR is the price of ignoring EV
   magnitudes — the first concrete measurement of the "magnitude-blind labels" risk.
3. **Identity inference is under-resolved while decision-relevant inference is fine.** The opponent head's
   KL to the exact posterior grows with context, and its entropy stalls at 3.9 nats. Two reasons, one to
   test: (a) the opponent-id loss was still decreasing at 12k steps (3.66 nats vs ln 256 = 5.55 prior), so the
   head is under-trained; (b) 256-way identity is the wrong target — many opponents are near-duplicates
   inside an archetype (κ = 30 around the Nash family) and are indistinguishable in 64 hands even for the
   exact posterior (2.3 nats left), yet they share the same best response. The policy does not need to
   know *who*, only *what to do*. E2 replaces the identity KL with the decision-relevant KL.
4. **No information seeking, by design.** The probes (bluff rate with J, call-down rate with Q) are flat
   from hand ~5 for every agent. None of Thompson, BayesBR or the DPT labels price information (§2.3);
   H3 needs the IDS or RL variants.
5. **Exploitation costs exploitability.** Every exploiter sits at 0.25–0.32 chips/hand exploitability,
   the same as the oracle best response itself; the equilibrium sits at 0. This is the trade-off H5 will
   put a knob on.
6. Seat effect: EV per hand alternates with seat (seat 0 is worse in Kuhn), so all curves are read per
   pair of hands.

## Decisions

- H1 as written was the wrong test and is retired in favour of a decision-relevant KL (see E2). The
  identity head stays as a diagnostic.
- H2's BayesBR comparison stays as the target; E2 asks whether Bayes-BR labels or longer training close
  the gap.
- Training budget 12k × 32 is sufficient for the policy but not for the identity head; E2 sweeps steps.

## Next: E2 (spec `docs/experiments/e2-decision-relevant-inference.md`)

Exact baselines PluralityBR and the policy-level KL, label variants (BR(θ) vs BR(posterior mixture)),
longer training and λ_opp sweep.
