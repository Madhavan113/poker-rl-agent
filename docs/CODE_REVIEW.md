# Code review policy

Every PR gets an **independent review**: performed by a person or agent who did not write the
code and works from the spec and the diff, reading the PR summary only for the claimed scope. The
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
5. Verdict, recorded on the PR in the form the `review-gate` check verifies against the live PR
   state (labels and comments, not the event payload):
   - **approve**: post a comment containing a line that is exactly
     `REVIEW-APPROVED <full 40-character head sha>` (the PR's head commit as shown by GitHub or
     `git rev-parse HEAD` on the reviewed checkout), then swap the labels: add `review: approved`
     and remove `review: changes requested`. Post the comment first; the label change re-runs the
     gate, which looks for the comment.
   - **changes requested**: the reverse. Post the findings as a comment, add
     `review: changes requested` and remove `review: approved`.
   The gate also requires the `ground-truth` label when the diff touches `truth/` or
   `tests/ground_truth/`. Any new commit removes `review: approved` and fails the gate; the
   re-review covers the delta, re-runs the preflight and ends with a new
   `REVIEW-APPROVED <new head sha>` comment.

## Checklist

Blocking categories:

- **Ground truth.** Any disagreement with `truth/`, or a test that was loosened, deleted or
  tolerances widened to pass.
- **Process files.** Any change under `.github/`, `scripts/`, `truth/` or `tests/ground_truth/`.
  Workflows run from the PR branch, so these files define the checks that judge the PR itself;
  read every changed line (see "Trust model" in `CONTRIBUTING.md`).
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
