# Code review policy

Every PR gets an **independent review**: performed by a person or agent who did not write the
code and has not seen the author's reasoning, working from the spec and the diff alone. The
reviewer runs `scripts/preflight.sh` themselves; a green CI is the floor, not the review.

## Reviewer procedure

1. Read the spec section(s) the PR claims to implement. Write down, before reading the code, what
   the implementation must satisfy.
2. Read the diff. For each public function, check name, signature and semantics against the spec.
3. Run the preflight locally. Then try to break the code: random-input identities, boundary
   cases, a hand-computed example the tests do not already contain.
4. Report findings ranked by severity. Each finding names file and line, states the defect in
   one sentence, and gives a concrete failing input or scenario. No style-only findings unless
   they hide a bug.
5. Verdict: **approve** (add label `review: approved`), or **changes requested** (add label
   `review: changes requested`, comment with the findings). Re-review after fixes covers the
   delta and re-runs the preflight.

## Checklist

Blocking categories:

- **Ground truth.** Any disagreement with `truth/`, or a test that was loosened, deleted or
  tolerances widened to pass.
- **Spec deviation** not declared in the PR.
- **Information leakage.** In this project the cardinal sin: an agent, tokenizer, or evaluator
  that can see the opponent's parameters, hidden cards, or the label at a point where the spec
  says it cannot. Trace where θ flows.
- **Wrong quantity.** A metric that computes something other than what the spec defines (EV of
  the wrong seat, exploitability without subtracting the game value, entropy in bits vs nats,
  Monte Carlo where the spec says exact).
- **Nondeterminism.** Unseeded randomness, dict ordering assumptions, floating tie-breaks that
  vary across platforms.
- **Silent failure.** Broad exception handling, fall-backs that hide missing data, NaNs that
  propagate into a plot.

Non-blocking but reported:

- Tests that only test the happy path or reuse the implementation to generate expectations.
- Hot-path performance where the spec calls it out (Kuhn engine, batched inference).
- Naming that will mislead a reader of the research (e.g. "regret" for something that is not).

## Research-specific items

- Plots carry error bars or bands with the number of sessions stated.
- Every reported number can be regenerated from a command line recorded in the PR.
- Baselines are run under exactly the same seeds and opponents as the method.
