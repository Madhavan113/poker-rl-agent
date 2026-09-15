# E2 — decision-relevant inference, label semantics, training convergence

Status: spec (2026-09-15). Follows the E1 findings in `e1-kuhn-results.md`.

## Questions

1. Does the transformer's action head converge to the DPT limit `q(a | h_t, I) = Σ_θ P(θ | h_t) 1[BR(θ)(I) = a]`?
   Measure it directly instead of through the identity head.
2. How much of the gap to BayesBR is label semantics (Thompson-style labels) versus under-training?
3. Does the identity head's KL gap close with more steps or a larger λ_opp, and does that change the policy?

## New exact quantities (evaluator additions)

- **Posterior over best-response actions** `π*_t(a | I) = Σ_i p_t(i) 1[BR(θ_i)(I) = a]` for every agent
  infoset `I` of the upcoming hand, from the exact posterior and the cached best responses.
- **Policy KL** `kl_policy[t] = Σ_I w_t(I) KL(π*_t(· | I) ‖ π_model(· | I))`, weighted by the agent's reach
  of each infoset under the *exact* opponent-free chance weights (uniform over the agent's card, then
  histories reachable under π*): reported per hand and averaged. This is the H1 replacement.
- **PluralityBR agent**: plays `argmax_a π*_t(a | I)` — the exact analogue of Transformer(argmax).
  Together with Thompson (exact analogue of Transformer(sample)) and BayesBR (myopic Bayes-optimal) this
  gives three exact references, one per label semantics.

## Conditions

| id | labels | steps × batch | λ_opp | purpose |
|----|--------|---------------|-------|---------|
| A (= E1) | BR(θ) | 12k × 32 | 0.5 | baseline |
| B | BR(θ) | 36k × 32 | 0.5 | convergence |
| C | BR(θ) | 12k × 32 | 0 | does the identity head help or hurt the policy |
| D | BR(θ) | 12k × 32 | 2 | forcing identity inference |
| E | BR(posterior mixture) | 12k × 32 | 0.5 | Bayes-BR labels: the myopic Bayes-optimal target |

Condition E needs the generator to compute the exact posterior along each session (the discrete E1a
population makes this exact and cheap: ~50 µs per hand) and label each decision with
`BR(mix(p_t))(I)`; a second exact reference, "BayesBR labels reached", is then `BayesBR` itself.

## Success criteria

- E2-1: `kl_policy` below 0.05 nats from t = 16 on, and Transformer(sample) within 0.01 chips/hand of Thompson
  and Transformer(argmax) within 0.01 of PluralityBR for t ≥ 16 (paired). Condition A is expected to fail this
  (E1 measured 0.018 mean / 0.030 max vs Thompson); the question is whether B (longer training) passes.
- E2-2: condition E is within 0.02 of BayesBR for t ≥ 16 (the H2 criterion E1 failed).
- E2-3: condition B lowers the identity KL below condition A's at every t; conditions C/D change
  Transformer EV by less than 0.01 (the identity head is a diagnostic, not a driver) — or they do not, which is
  itself the finding.

## Budget

Five training runs ≈ 1 + 3 + 1 + 1 + 1 hours on the M4 (MPS); evaluation ≈ 30 min each with the current
evaluator. Run B first in the background; A is done.
