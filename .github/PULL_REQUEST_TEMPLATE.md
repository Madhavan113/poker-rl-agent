## Summary

<!-- What and why, 2–5 sentences. One concern per PR. -->

## Spec conformance

- Spec section(s): <!-- e.g. docs/experiments/e1-kuhn.md §Interfaces -->
- Deviations: none <!-- or list each, with justification; add the `spec-deviation` label -->

## Ground truth

<!-- Paste the PREFLIGHT OK line printed by scripts/preflight.sh for the HEAD commit. The
     ground-truth CI job fails unless its commit= is a prefix of the PR head and its truth= equals
     scripts/check_truth.py --hash at that commit: re-run the preflight and replace this line after
     every push. -->

```
PREFLIGHT OK ...
```

## Tests

- New or changed tests:
- What would have caught a regression here that the existing tests did not?

## Review

- [ ] Independent review completed per docs/CODE_REVIEW.md; findings linked below and resolved.
      The author ticks this box only after the reviewer's `REVIEW-APPROVED <head sha>` comment
      exists on this PR, never in advance.
- [ ] No new runtime dependencies (or justified in Summary)
- [ ] No data, checkpoints or `runs/` output committed
- [ ] Seeds and configs recorded for any reported number

Findings: <!-- link to the review comment(s) -->
