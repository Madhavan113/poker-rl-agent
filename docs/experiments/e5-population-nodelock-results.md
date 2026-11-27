# E5 results — population node-locking against static and adaptive opponents (Leduc)

Run of 2026-11-26 (`runs/e5/`; command line, config and commit in `runs/e5/README.md`, code at
`d4e34dc`). Everything is exact except the observations: the EV of each block is our block strategy
against the opponent's current profile by tree traversal, averaged over seats. ± is one standard error
over 64 opponents per set. Every config sees the same deals (seeded per opponent, block and hand).
Sessions are 16 blocks × 32 hands. Re-solves are cold, 300 CFR+ iterations each. The largest certified
duality gap of any re-solve is 0.0096 chips/hand (mean ≤ 0.0044), so solver error is below every
difference discussed here. Wall time: 1 h 39 min on the M4 (8 workers).

"Gain" is the paired EV minus the equilibrium agent's EV against the same opponents. "Late" is the
second half of the session (blocks 8–15). Methods (spec §Agent protocol):

- `br`: full lock, i.e. the best response to the inferred opponent.
- `hard`: lock only infosets with n_I ≥ 10.
- `soft`: lock confidence c_I = n_I / (n_I + 10).
- `rnr_lo`, `rnr_hi`: restricted Nash response with p = 0.5 and p = 0.8.

Inference: `bayes` is the population posterior; `em` is the prior-free MAP. γ is the per-block
forgetting factor.

## Static opponents (η = 0), in-population

Nash earns +0.068. The static oracle (best response knowing θ) earns +0.602, a ceiling of +0.534 over
Nash.

| method | gain, bayes | gain, em | exploitability (bayes / em) |
|---|---|---|---|
| br | **+0.489 ± 0.039** | +0.211 ± 0.024 | 2.77 / 2.65 |
| rnr_hi (p = .8) | +0.365 ± 0.039 | +0.183 ± 0.026 | 0.47 / 0.22 |
| rnr_lo (p = .5) | +0.194 ± 0.031 | +0.120 ± 0.020 | 0.07 / 0.04 |
| soft | +0.032 ± 0.006 | +0.026 ± 0.004 | 0.01 / 0.01 |
| hard | +0.029 ± 0.003 | +0.024 ± 0.003 | 0.20 / 0.22 |

γ = 1 in this table; γ = 0.8 is within 0.02 for every bayes row and slightly lower for em.

The full lock on the population posterior captures 92% of the oracle's edge after one block of
data. RNR(0.5) keeps 36% of it at 1/40th of the exploitability. Fresh opponents, whose profiles the
population does not contain, give the same picture slightly lower (`br` +0.426, `rnr_hi` +0.324,
`rnr_lo` +0.191 with bayes).

## Adaptive opponents

The opponent moves by η toward a best response to the block we just played; it sees that strategy
exactly. Nash's own EV falls as the opponents adapt (+0.068, +0.035, +0.014 at η = 0, 0.1, 0.3),
because a best response to Nash plugs the opponent's leaks without being able to exploit Nash.

In-population, session gain over Nash (late gain in brackets), best inference/γ per method:

| method | η = 0.1 | η = 0.3 |
|---|---|---|
| br (bayes, γ = .8) | −0.221 ± 0.019 (−0.356) | −0.639 ± 0.018 (−0.935) |
| rnr_hi (bayes, γ = .8) | +0.071 ± 0.010 (−0.001) | −0.026 ± 0.003 (−0.074) [em, γ = .8] |
| rnr_lo (bayes, γ = 1) | **+0.069 ± 0.011** (+0.030) | +0.036 ± 0.002 (+0.009) [em, γ = .8] |
| rnr_lo (em, γ = .8) | +0.061 ± 0.006 (+0.052) | **+0.036 ± 0.002** (+0.009) |
| soft (em, γ = .8) | +0.015 ± 0.001 (+0.019) | +0.013 ± 0.000 (+0.011) |
| hard (em, γ = .8) | +0.012 ± 0.001 (+0.019) | +0.009 ± 0.001 (+0.011) |

The worst `br` config loses 0.73 chips/hand to Nash over the session at η = 0.3 (bayes, γ = 1),
and 1.12 in the second half on fresh opponents. Its curve (`ev_by_block_*.png`) peaks at block 1 and
falls below Nash by block 3. `rnr_hi` follows `br` down more slowly and ends below Nash at η = 0.3.
`rnr_lo` stays at or above Nash in every condition: its worst late gain is −0.038 (bayes, γ = 1,
fresh, η = 0.3). `hard` and `soft` never move far from Nash in either direction (|late gain| ≤ 0.05
everywhere). Fresh opponents give the same ordering everywhere; `br`'s losses are larger on them (late −1.12 vs
−0.98 at η = 0.3, bayes, γ = 1), and every other row is within 0.04.

## Hypotheses

| | verdict | reading |
|---|---|---|
| E5-1 static ranking br ≥ rnr_hi ≥ soft ≥ hard ≥ nash | **PASS** | holds for both inferences; soft and hard are within one s.e. of each other |
| E5-2 at η = 0.3, br's late gain < 0; best late gain is an RNR config | **PARTIAL** | br late −0.93 to −1.12: pass. Best late gain is soft/hard em γ = .8 (+0.011) with rnr_lo em γ = .8 at +0.009: a tie at about Nash, not an RNR win. Over the whole session rnr_lo is best (+0.036). |
| E5-3 forgetting helps br more than RNR under adaptation | **FAIL** | true only for bayes at η = 0.1 (br late −0.682 → −0.356). For em, forgetting makes br slightly worse and RNR slightly better; at η = 0.3 it helps br and rnr_hi about equally. |
| E5-4 bayes beats em (in-pop), gap shrinks on fresh | **PASS** at η = 0 | bayes > em for all five methods, by 0.005 (hard) to 0.278 (br); the gaps shrink by 0–0.04 on fresh opponents. Under adaptation the order often reverses (em γ = .8 is best for rnr_lo late at both η). |
| E5-5 hard/soft within 0.05 of Nash; RNR buys robustness more cheaply | **PASS** | max \|gain\| 0.032 for hard/soft; rnr_lo gets 6× soft's static gain at exploitability 0.07 |

## Findings

1. **The money is in the population model, not in which nodes you lock.** Against static opponents
   the best response to the population posterior earns 8× Nash's EV (+0.557 vs +0.068). The same lock fitted without the
   population (EM) earns less than half as much. Partial node locks earn almost nothing: Leduc has
   468 opponent infosets per seat, and after 480 hands only about 2% of them have 10 or more
   expected observations (measured on 8 in-population opponents, Nash collector). Unlocked nodes let the solver's adversary undo the lock (the "re-route the range"
   effect from the spec), so a lock only pays where it covers the opponent's whole decision path. The
   population posterior gets around this by locking the whole tree through a low-dimensional leak
   model. That is the version of "node-lock an unbalanced range" that works.
2. **Full locking is a trap against anyone who adapts.** A 10% move toward a best response every 32
   hands turns +0.49 into −0.40 per hand (bayes, γ = 1). Exploitability 2–3 chips/hand is the bill
   coming due: the adapter only has to find the best response to a pure strategy.
3. **RNR(0.5) is the robust choice in this grid.** It never falls meaningfully below Nash (worst late
   −0.038) and wins in every static and slow-adaptation condition: +0.19 static, +0.07 at η = 0.1.
   RNR(0.8) is better static (+0.37) and worse adaptive (it ends below Nash at η = 0.3). The right p
   depends on how fast the opponent adapts. That is the open problem this experiment exposes.
4. **Against a fast omniscient adapter nothing beats Nash by much.** At η = 0.3 the best session gain
   is +0.036 and the best late gain about +0.01. This adapter is the worst case of its family (it
   reads our strategy exactly); a realistic opponent who estimates it from observations is slower.
5. **Forgetting is not a fix.** γ = 0.8 rescues a full lock only partly and only with the population
   prior. Prior-free estimates with forgetting are too noisy to best-respond to.

## Decisions and next steps

- Keep RNR as the exploitation knob. Hard and soft node locks stay in the solver as baselines, not as
  methods.
- **E5b: choose p online.** For example, safe exploitation (Ganzfried & Sandholm 2015): risk only
  what has been won so far, and lower p when realized EV falls below the model's prediction (a direct
  signal of adaptation). Target: rnr_hi's static gain together with rnr_lo's robustness.
- **E5c: realistic adapters** that estimate our strategy from what they observe, with a lag; plus
  opponents outside the prior family.
- Prior-free inference at the leak-model level (fit the few archetype parameters, not 468 infosets
  per seat) to close the bayes–em gap without assuming the population.
- Later: the E1/E2 transformer as the inference engine feeding the same locked solver.
