# exsolver

Research code for an **exploitative poker solver**: a small transformer that, given the live
history of a session, infers a posterior over the opponent's strategy in context and best-responds
to it. See `RESEARCH.md` for the research direction and `docs/experiments/` for experiment specs.

Layout

- `src/exsolver/games/`      extensive-form game engines (Kuhn, Leduc) with a common interface
- `src/exsolver/solvers/`    CFR+ equilibrium, exact best response, exploitability, strategy mixing
- `src/exsolver/population/` opponent priors (the dataset *is* the prior)
- `src/exsolver/bayes/`      exact Bayesian posterior over a discrete opponent population
- `src/exsolver/agents/`     equilibrium, oracle, Thompson, Bayes-BR, transformer agents
- `src/exsolver/data/`       tokenizer, session generation (DPT recipe), shard format
- `src/exsolver/model/`      decoder-only transformer with action head and opponent head
- `src/exsolver/eval/`       exact per-hand metrics, plots
- `src/exsolver/experiments/` runnable experiments (E1: Kuhn)

Tooling: `uv sync`, `uv run pytest`, `uv run ruff check`.
