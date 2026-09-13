# Ground truth

Values the code must reproduce, derived **independently of the code**. Consumed by
`tests/ground_truth/` (marker `ground_truth`), validated by `scripts/check_truth.py`, run by
`scripts/preflight.sh` and the `ground-truth` CI job.

Statuses: `verified` (analytic or reproduced from an external reference with version recorded),
`corroborated` (external reference recalled, independent run pending), `unverified`.

## Kuhn derivations (`kuhn.json`)

Deals are uniform over the 6 ordered pairs of distinct cards; a player's card is J, Q or K with
probability 1/3 and the opponent's card is each of the other two with probability 1/2.

- **Game value** −1/18 for seat 0 and the one-parameter Nash family are Kuhn (1950).
- **Seat-1 station** (bets every card after a check, calls every bet). Seat 0 bets and is always
  called (showdown for 2): J −2, Q 0, K +2. Seat 0 checks and is always bet into: calling gives
  the same, folding −1. Best response: J check–fold −1, Q 0, K +2 → (−1 + 0 + 2)/3 = 1/3.
- **Seat-0 maniac** (bets every card). Seat 1 facing a bet: J fold −1 (call −2), Q call 0,
  K call +2 → 1/3.
- **Seat-1 folder** (never bets after a check, folds every bet). Seat 0 bets every card and wins
  the ante: +1 always → 1.

## Leduc (`leduc.json`)

OpenSpiel's `leduc_poker` (6 cards, antes 1, bets 2 then 4, at most two raises per round) has
936 information states and a seat-0 value of about −0.0856 under CFR. Status `corroborated` until
an independent OpenSpiel run is recorded here with its version.
