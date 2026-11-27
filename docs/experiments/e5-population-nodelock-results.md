# E5 results — population node-locking against static and adaptive opponents (Leduc)

Run of 2026-11-26 (`runs/e5/`). The command line, config and commit are in `runs/e5/README.md`; the
run is at `d4e34dc`, and summary and figures were rebuilt with `--reaggregate` at `529f4ea`. The
rebuild changed no number; it adds standard errors per archetype and error bars on the frontier.

Everything is exact except the observations. The EV of each block is our block strategy against the
opponent's current profile, by tree traversal, averaged over the two seats. ± is one standard error
over 64 opponents per set. Every config sees the same deals, seeded per opponent, block and hand
(`play_block`). Sessions are 16 blocks × 32 hands. Re-solves are cold, 300 CFR+ iterations each. The
largest certified duality gap of any re-solve is 0.0096 chips/hand, and the mean gap per config is at
most 0.0044. Wall time: 1 h 39 min on the M4 with 8 workers.

**Definitions.**
- *Gain*: the paired EV minus the equilibrium agent's EV against the same opponents.
- *Session gain*: averaged over all 16 blocks, including block 0, which every agent plays as Nash.
- *Late*: the second half of the session, blocks 8–15.

**Methods** (spec §Agent protocol):
- `br`: full lock, i.e. the best response to the inferred opponent.
- `hard`: lock only infosets with n_I ≥ 10.
- `soft`: lock with confidence c_I = n_I / (n_I + 10).
- `rnr_lo` and `rnr_hi`: the restricted Nash response with p = 0.5 and p = 0.8.

**Inference:**
- `bayes`: the population posterior.
- `em`: the prior-free posterior mode.

γ is the per-block forgetting factor.

## Static opponents (η = 0), in-population

Nash earns +0.068. The static oracle, a best response that knows θ, earns +0.602, so the ceiling is
+0.534 over Nash.

| method | session gain, bayes | session gain, em | exploitability (bayes / em) | gain per unit of exploitability (bayes / em) |
|---|---|---|---|---|
| br | **+0.489 ± 0.039** | +0.211 ± 0.024 | 2.77 / 2.65 | 0.18 / 0.08 |
| rnr_hi (p = .8) | +0.365 ± 0.039 | +0.183 ± 0.026 | 0.47 / 0.22 | 0.78 / 0.85 |
| rnr_lo (p = .5) | +0.194 ± 0.031 | +0.120 ± 0.020 | 0.07 / 0.04 | 2.82 / 3.24 |
| soft | +0.032 ± 0.006 | +0.026 ± 0.004 | 0.01 / 0.01 | 2.97 / 2.39 |
| hard | +0.029 ± 0.003 | +0.024 ± 0.003 | 0.20 / 0.22 | 0.14 / 0.11 |

The table uses γ = 1. With γ = 0.8, the bayes rows are within 0.003 for br and both RNRs, but soft
falls 0.014 and hard 0.021. Forgetting starves the partial locks of observations.

- **Full lock.** On the population posterior it captures 92% of the oracle's edge over the session
  (0.489 / 0.534), including the Nash block 0. The same lock fitted without the population earns 43%
  of that gain.
- **RNR(0.5)** keeps 36% of the oracle edge at 1/40th of the full lock's exploitability.
- **Soft and hard locks** earn little (about 6% of the ceiling). Soft spends almost no exploitability
  doing it, so per unit of risk it is about as efficient as RNR(0.5). Hard is dominated: it pays
  RNR-level exploitability for soft-level gain.
- **Fresh opponents** (profiles the population does not contain) give the same ranking with bayes:
  br +0.426, rnr_hi +0.324, rnr_lo +0.191, soft +0.030, hard +0.026. With em, rnr_hi and br swap
  places (+0.184 vs +0.173, within one s.e.).

## Adaptive opponents

After each block, the opponent moves by η toward a best response to the block we just played, and it
sees that strategy exactly. Nash's own EV falls as the opponents adapt (+0.068, +0.035 and +0.014 at
η = 0, 0.1 and 0.3). A best response to Nash plugs the opponent's leaks but cannot exploit Nash.

The table shows in-population session gain over Nash, with late gain in brackets, for the
best-performing inference and γ per method and η:

| method | η = 0.1 | η = 0.3 |
|---|---|---|
| br | −0.221 ± 0.019 (−0.356) [bayes, γ = .8] | −0.639 ± 0.018 (−0.935) [bayes, γ = .8] |
| rnr_hi | +0.071 ± 0.010 (−0.001) [bayes, γ = .8] | −0.026 ± 0.003 (−0.074) [em, γ = .8] |
| rnr_lo | **+0.069 ± 0.011** (+0.030) [bayes, γ = 1] | **+0.036 ± 0.002** (+0.009) [em, γ = .8] |
| soft | +0.019 ± 0.001 (+0.024) [em, γ = 1] | +0.013 ± 0.000 (+0.011) [em, γ = .8] |
| hard | +0.022 ± 0.002 (+0.028) [em, γ = 1] | +0.009 ± 0.001 (+0.011) [em, γ = .8] |

- **Full lock.** It peaks at block 1 and falls below Nash at block 2 (η = 0.3) or block 3 (η = 0.1);
  see `ev_by_block_*.png`. Its worst session gain is −0.762 ± 0.021 (fresh, η = 0.3, bayes, γ = 1).
  In-population the worst is −0.732. At η = 0.3 its late gain ranges from −0.92 to −1.12 across
  configs.
- **rnr_hi** follows br down more slowly and ends below Nash at η = 0.3.
- **rnr_lo.** Its session gain is ≥ 0 in every condition; the minimum is +0.003, fresh, η = 0.3. Its
  late gain is below Nash in 6 of its 8 η = 0.3 configs, though, and the worst is −0.038 ± 0.005
  (fresh, bayes, γ = 1).
- **Soft and hard** never move far from Nash in either direction: max |session gain| 0.032, max |late
  gain| 0.046. At η = 0.3, soft's late gain is at least rnr_lo's in all 8 configs.
- **Fresh opponents** reproduce the non-br rows within 0.04 at η > 0. br loses more on them (late
  −1.12 vs −0.98 at η = 0.3, bayes, γ = 1).

## Hypotheses

| | verdict | reading |
|---|---|---|
| E5-1: static ranking br ≥ rnr_hi ≥ soft ≥ hard ≥ nash | **PASS** | holds for both inferences at γ = 1 (soft and hard within one s.e. there); at γ = 0.8 too |
| E5-2: at η = 0.3, br's late gain < 0, and the best late gain is an RNR config | **FAIL** (first clause passes) | br late −0.92 to −1.12. The best late gains are soft and hard (em, γ = .8, both +0.011 ± 0.001), slightly above rnr_lo (+0.009 ± 0.002): all about Nash. rnr_lo has the best *session* gain (+0.036). |
| E5-3: forgetting helps br more than RNR under adaptation | **FAIL** | true only for bayes at η = 0.1 (br late −0.682 → −0.356). At η = 0.3 with bayes it helps br and rnr_hi about equally. With em, forgetting makes br slightly worse and RNR slightly better at both η. |
| E5-4: in-pop, bayes beats em for every method; the gap shrinks on fresh | **PARTIAL** | at η = 0, γ = 1: bayes > em for all five methods, by 0.005 (hard) to 0.278 (br), and every gap shrinks on fresh (by 0.0005 to 0.042). Under adaptation em is often better: br, hard and soft at η = 0.1; the best soft/hard/rnr_lo configs at η = 0.3 are em. |
| E5-5: hard/soft within 0.05 of Nash; RNR buys robustness more cheaply | **PARTIAL** | first clause passes (max \|gain\| 0.032). RNR beats hard on gain and on exploitability. It does not beat soft: per unit of exploitability they are equal (2.82 vs 2.97, bayes), and at η = 0.3 soft's late gain is at least rnr_lo's everywhere. |

## Findings

1. **The money is in the population model, not in which nodes you lock.** Against static opponents,
   the best response to the population posterior earns 8× Nash's EV (+0.557 vs +0.068). The same lock
   fitted without the population earns 43% of that gain.
   - **Why per-node locks starve.** Leduc has 468 opponent infosets per seat, since infosets keep
     suits. After 480 hands only about 2.4% of them have 10 or more expected observations. (Measured
     with `expected_visits` under the true θ on in-population opponents b = 0, 8, …, 56, playing the
     equilibrium, seeds `default_rng([9, b, h])`; independently reproduced in review.)
   - **Why unlocked nodes cost.** Unlocked nodes let the solver's adversary undo the lock (the
     "re-route the range" effect in the spec), so a partial lock is worth little.
   - **What works.** The population posterior locks the whole tree through a low-dimensional leak
     model, which is the version of "node-lock an unbalanced range" that works.
2. **Full locking is a trap against anyone who adapts.** A 10% move toward a best response every 32
   hands turns +0.49 into −0.40 per hand (bayes, γ = 1). The full lock's exploitability of 1.7–2.8
   chips/hand is the bill coming due, since the adapter only has to find the best response to a pure
   strategy.
3. **Three usable points on the frontier, and no single best one.**
   - **Full lock.** The most EV against static opponents and a disaster against adaptive ones.
   - **RNR(0.5).** The most session EV once anyone adapts: +0.069 at η = 0.1, tied with rnr_hi's
     +0.071 ± 0.010 there, and the best at η = 0.3 (+0.036). Still +0.19 static. It drifts slightly below Nash late against fast adapters
     (worst −0.038).
   - **Soft lock.** The same gain per unit of risk at a tenth of the scale (+0.03). It never strays
     from Nash by more than 0.05.
   - **RNR(0.8).** Better static (+0.37), but worse once opponents adapt.

   The right trust depends on how fast the opponent adapts, which is the open problem this
   experiment exposes.
4. **Against a fast omniscient adapter nothing beats Nash by much.** At η = 0.3 the best session gain
   is +0.036 and the best late gain about +0.01. This adapter is the worst case of its family, because
   it reads our strategy exactly. A realistic opponent who must estimate it from observations is
   slower.
5. **Forgetting is not a fix.** γ = 0.8 rescues a full lock only partly, and only with the population
   prior. It costs the partial locks observations in the static case.

## Decisions and next steps

- RNR is the main exploitation knob and soft locks are the conservative option. Hard locks are
  dominated in every condition here and stay only as a baseline.
- **E5b: choose p online.** For example, safe exploitation (Ganzfried & Sandholm 2015): risk only
  what has been won so far, and lower p when realized EV falls below the model's prediction, which is
  a direct signal of adaptation. Target: rnr_hi's static gain with soft's worst case. A soft lock
  with an adaptive confidence is a second candidate.
- **E5c: realistic adapters** that estimate our strategy from what they observe, with a lag, plus
  opponents outside the prior family.
- **Prior-free inference at the leak-model level:** fit the few archetype parameters instead of 468
  infosets per seat, to close the bayes–em gap without assuming the population.
- **Next run:** a seed tag that avoids the trailing-zero collision documented in `play_block`.
- **Later:** the E1/E2 transformer as the inference engine feeding the same locked solver.
