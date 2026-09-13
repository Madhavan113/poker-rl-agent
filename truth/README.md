# Ground truth

Values the code must reproduce, derived **independently of the code**. Consumed by
`tests/ground_truth/` (marker `ground_truth`), validated by `scripts/check_truth.py`, and run by
`scripts/ground_truth.sh`, the one command behind both `scripts/preflight.sh` and the
`ground-truth` CI job.

## Statuses and schema

`status` is exactly one of:

- `verified`: an analytic result, or a number reproduced from an external reference
  implementation with its version recorded and the run's script and output committed under
  `truth/derivations/`.
- `unverified`: everything else (recalled, estimated, or awaiting an independent run).

Every file needs `game`, `status`, `provenance` (non-empty list of strings) and `tolerances`
(dict). Schema rule enforced by `scripts/check_truth.py`: an `unverified` file must not contain
`game_value_seat0` or `hand_computed_best_responses`, because the ground-truth tests enforce
those numerically; a value the code is held to must have verified provenance.

`scripts/check_truth.py --hash` prints sha256[:12] over the sorted concatenation of
`truth/*.json`. That is the `truth=` value in the `PREFLIGHT` line and the hash CI recomputes
against the PR body; the table printed without `--hash` uses the same function per file, so only
one kind of truth hash exists.

## Pending engine modules

`tests/ground_truth/pending_engine.txt` lists, one dotted module name per line (blank lines and
lines starting with `#` are ignored), the engine modules the ground-truth tests import but that
do not exist yet. `require_module` in `tests/conftest.py` skips a test module, visibly under
`-rs`, only when the missing module is listed; a missing module that is not listed fails
collection. **The engine PR must delete the entries it implements**; the file may be removed once
it is empty. When the file is empty or absent, `scripts/ground_truth.sh` adds `--forbid-skips`,
which turns every skip into a failure, so the suite cannot pass vacuously once the engine exists.

## Kuhn derivations (`kuhn.json`, verified)

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

## Leduc (`leduc.json`, verified)

Reference: OpenSpiel 2.0.2, game `leduc_poker` with default parameters (`players=2`,
`starting_player=0`, `suit_isomorphism=False`), run independently on 2026-09-13. Script
`derivations/openspiel_leduc.py`, output `derivations/openspiel_leduc.out`. Reproduce with

```
uv run --isolated --no-project --python 3.12 --with open_spiel==2.0.2 --with numpy python truth/derivations/openspiel_leduc.py
```

Results: 936 information states, 5520 terminal histories, max utility 13; CFR+ average-policy
value for seat 0 −0.08559349 after 1000 iterations and −0.08560604 after 5000 iterations, with
exploitability 1.8385e−05 (NashConv/2). `game_value_seat0` is the 5000-iteration value. The
test runs 1000 iterations with tolerances `cfr_value` 5e−4 and `cfr_exploitability` 1e−3;
OpenSpiel's own 1000-iteration numbers sit 1.3e−5 and 2.6e−4 inside those bounds.

Rules, spelled out in the file's `rules` field: 6 cards = 3 ranks × 2 suits; ante 1; seat 0
acts first in both rounds; bet/raise size 2 in round 1 and 4 in round 2; at most two raises per
round with the opening bet counting toward the cap; fold only when facing a bet; one public card
before round 2; a pair with the board beats a high card, otherwise the higher rank wins; equal
ranks split the pot. Card index = rank·2 + suit (Js=0, Jh=1, Qs=2, Qh=3, Ks=4, Kh=5).

Hand check of the terminal count: each betting round has 5 sequences that end without a fold
(`cc`, `cbc`, `cbbc`, `bc`, `bbc`) and 4 that end in a fold (`cbf`, `cbbf`, `bf`, `bbf`). Per
ordered deal (30 of them): 4 round-1 folds + 5 × 4 board cards × 9 round-2 endings = 184;
30 × 184 = 5520.
