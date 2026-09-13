# Exploitative solver — research direction

Started 2026-09-13. Owner: Madhavan. Code in this directory; experiment specs in `docs/experiments/`.

## 0. Thesis in one paragraph

Equilibrium poker AIs (Libratus, Pluribus, DeepStack, ReBeL, Student of Games) play one fixed,
near-unexploitable strategy and leave every opponent-specific chip on the table. We want the
opposite object: an agent that treats each observation in a session as evidence about *this*
opponent, maintains a belief over their strategy, and best-responds to that belief while pricing
the information it still lacks and the exploitability it is taking on. The central technical
claim is that this is not primarily an architecture problem. Against a prior over opponents the
optimal policy is Bayes-optimal play, and a sequence model trained by maximum likelihood on
sessions sampled from that prior learns Bayes-optimal in-context play for free (Ortega et al.
2019; Lee et al. 2023). What remains, and what no prior work solves together, are three
problems: **the prior** (which opponent distribution we train on), **information pricing** (when
to pay chips to reduce uncertainty), and **safety** (how exploitable we let ourselves become).
The programme below attacks them in a game where every quantity is exactly computable, then
scales.

## 1. Setting

Two-player zero-sum extensive-form game with imperfect information, played for `T` hands with
alternating seats. The opponent plays a behavioural strategy θ over *both* seats. Phase 0 assumes
θ is fixed for the session; non-stationarity is Phase 2. The agent observes `h_t`: its own cards,
the board, every action by both players, showdown reveals, and results. Nothing else.

## 2. Formalism

### 2.1 Objective and belief

    maximise over π:   E_{θ ~ P}  E[ Σ_{t=1..T} u_t | π, θ ]

This is a Bayes-adaptive POMDP whose sufficient statistic is the belief `b_t = P(θ | h_t)`.
Everything else in this document is about representing `b_t`, acting on it, and shaping `P`.

### 2.2 The posterior, and why our own actions do not bias it

For a single hand in which the agent held card `c_me` and the opponent took actions
`a_1..a_k` at histories `I_1..I_k`:

    L(hand | θ) = Σ_{c_opp} P(c_opp | c_me, board) · Π_k θ(a_k | I_k(c_opp))

The sum over the opponent's *unknown* private card is joint across all their actions in the hand
(they share one card). If a showdown revealed `c_opp`, the sum collapses to one term. The agent's
own actions enter the likelihood of the observed sequence only as factors that do not depend on
θ, so `P(θ | h_t)` is **invariant to the policy the agent used to generate `h_t`**. Our policy
decides which observations *exist* (fold and there is no showdown; check and the opponent may
bet), not how existing observations are interpreted. Consequence for training: any mixture of
data-collection policies yields correct posteriors; the mixture only needs to cover the contexts
the deployed agent will create for itself. This is the property that makes the DPT recipe below
safe under collection-policy shift.

### 2.3 Information and its price

Entropy `H(b_t)` measures uncertainty about θ. The expected information gain of an action is
`g_t(a) = E[H(b_t) − H(b_{t+1}) | a]`. Two corrections to the naive "reduce entropy every step":

1. **Entropy is not the objective.** Only decision-relevant uncertainty has value. An agent that
   minimises entropy will call every bet to see cards and lose money doing so.
2. **The right trade-off is information-directed.** Information-Directed Sampling (Russo & Van
   Roy) picks the action minimising `Δ_t(a)² / g_t(a)`, where `Δ_t(a)` is the expected shortfall of
   `a` against the oracle best response under the current belief. Poker gives `g_t` concrete
   prices: calling to reach showdown, checking to let the opponent bet, raising to see how they
   react to aggression. Uncertainty about rarely reached nodes never resolves unless we pay to
   reach them.

Useful bounds. The total value that information can ever add is
`Σ_t [EV(oracle BR) − EV(Bayes-myopic_t)]`, which shrinks as the posterior concentrates; so
exploration is worth most early in a session and in the tails of the game tree. For a discrete
population the posterior mass on wrong opponents decays exponentially at a rate set by the
Chernoff information between the observation distributions they induce; for a continuous
parametrisation, posterior variance falls like `1/t` and differential entropy like
`−(d/2) log t`. Both shapes are testable in E1.

### 2.4 On the "Hilbert space" question

The opponent's strategy lives in a finite-dimensional convex polytope (sequence form: a product
of simplices, one per information set). A belief is a probability measure on that polytope. The
polytope is astronomically large for real poker, so the agent needs a low-dimensional latent in
which beliefs live and shrink. Three statements make this precise:

- The natural geometry on strategy space is **Fisher–Rao**, not Euclidean: the distance between
  two opponents that matters is the KL divergence between the action distributions they induce
  on the nodes we actually reach. Two opponents identical at reached nodes are the same opponent.
- The **hidden state of a sequence model trained on the prior is a sufficient statistic of the
  posterior** (Ortega et al. 2019). The latent is learned, not designed. It is a Hilbert space
  only in the trivial sense that it is a vector space with an inner product; its useful structure
  is that linear read-outs of it recover posterior quantities.
- We make that read-out **explicit and measurable**: an auxiliary head predicts the opponent's
  identity (discrete population) or parameters (continuous prior). Its predictive entropy is the
  quantity we plot, and in Kuhn we compare it against the exact posterior.

### 2.5 Amortised Bayesian play: the DPT recipe

Train a causal transformer on sessions sampled from `P`. Input: the token stream of the session.
At each of the agent's decision points the target is the **best-response action against the true
sampled θ**, computed offline with full knowledge of θ. By the meta-learning argument the
predictive distribution converges to

    q(a | h_t, I) = Σ_θ P(θ | h_t) · 1[ BR(θ)(I) = a ]

the posterior over best responses. Sampling from `q` is posterior sampling (Thompson) at the
action level; the argmax is the posterior-plurality best response. Known limitations, both
measurable in Kuhn against exact-Bayes agents: `q` ignores EV magnitudes (a 51/49 posterior
between two best responses with very different stakes is treated as a coin flip) and it does not
price information. Alternatives to test in E2: label with the best response to the posterior
*mixture* (the myopic Bayes-optimal action, exact in a discrete population), label with the IDS
action, or fine-tune with RL on cumulative EV so information value is learned implicitly.

### 2.6 Safety

Exploitability of the agent's hand-`t` strategy, `ε(π_t) = max_θ' EV_θ'(π_t) − (−v*)`, is the
price of commitment. Ganzfried & Sandholm's safe exploitation keeps cumulative worst-case loss
below cumulative gain over the equilibrium value; Restricted Nash Response mixes the
best-response with equilibrium at the strategy level. In Phase 0 we only *measure* `ε(π_t)`; in
Phase 1 we add a KL-to-equilibrium penalty or RNR mixing and map the exploitation/exploitability
Pareto frontier.

### 2.7 Non-stationarity

Real opponents adapt, so θ drifts and posterior entropy can rise. Model it in the prior as drift
or changepoints; the sequence model then learns a forgetting posterior. A copy of the agent as an
opponent gives mutual-exploitation dynamics worth studying on their own. Phase 2.

## 3. The prior is the dataset

Because the trained model is Bayes-optimal *for whatever distribution generated its training
sessions*, designing the opponent population is the central research act.

### 3.1 Kuhn prior (Phase 0, concrete)

An opponent is θ ∈ [0,1]^12: for each of the 12 information sets, the probability of the
aggressive action (bet, or call when facing a bet). Parameter order and centres:

| infoset (seat:card|history)     | equilibrium(α) | maniac | station | rock | random |
|--------------------------------|----------------|--------|---------|------|--------|
| 0:J|   P(bet)                   | α              | .70    | .05     | .02  | U(0,1) |
| 0:Q|   P(bet)                   | 0              | .80    | .15     | .05  | U(0,1) |
| 0:K|   P(bet)                   | 3α             | .95    | .80     | .90  | U(0,1) |
| 0:J|cb P(call)                  | 0              | .50    | .60     | .02  | U(0,1) |
| 0:Q|cb P(call)                  | α + 1/3        | .85    | .95     | .20  | U(0,1) |
| 0:K|cb P(call)                  | 1              | 1.0    | 1.0     | .95  | U(0,1) |
| 1:J|c  P(bet)                   | 1/3            | .70    | .05     | .02  | U(0,1) |
| 1:Q|c  P(bet)                   | 0              | .70    | .10     | .05  | U(0,1) |
| 1:K|c  P(bet)                   | 1              | .95    | .90     | .95  | U(0,1) |
| 1:J|b  P(call)                  | 0              | .50    | .60     | .02  | U(0,1) |
| 1:Q|b  P(call)                  | 1/3            | .90    | .95     | .15  | U(0,1) |
| 1:K|b  P(call)                  | 1              | 1.0    | 1.0     | 1.0  | U(0,1) |
| **mixture weight**              | **.30**        | **.15**| **.20** | **.20**| **.15**|
| **Beta concentration κ**        | **30**         | **10** | **10**  | **10** | (1,1)  |

Each parameter is drawn `Beta(m·κ, (1−m)·κ)` around its archetype centre `m` (clipped to
[.02, .98]); for the equilibrium archetype `α ~ U[0, 1/3]` is drawn per opponent. Rationale:
the equilibrium mass teaches the agent to fall back to near-GTO when it sees nothing exploitable;
maniac / station / rock are the canonical human leak clusters (over-aggression, over-calling,
over-folding); the random component guarantees every observation has non-zero likelihood under
the prior so posteriors never collapse on a lie. Two instantiations: **E1a** freezes a population
of 256 opponents drawn once from this prior, which makes the exact Bayesian posterior computable
for comparison; **E1b** draws a fresh θ per session and tests generalisation. The population
summary (`runs/e1/population_summary.md`) reports, per archetype, how much an oracle best
response wins over equilibrium — the exploitable value that the in-context agent is trying to
capture.

### 3.2 Session data distribution

    opponent  ~ prior (or uniform over the frozen population)
    collector ~ {equilibrium .4, uniform-random .3, oracle BR(θ) .3}   (one per session)
    play H = 64 hands, alternating seats; opponent acts from θ, agent from the collector
    tokens: BOS, then per hand  HAND POS CARD actions… (SHOW_c | NO_SHOW) RESULT END_HAND
    labels: BR(θ) action at every agent decision; opponent id / θ at every hand boundary

### 3.3 Priors for larger games (Phases 1–3)

- **Perturbed CFR**: solve the game (or an abstraction) with CFR+, then perturb in logit space
  with archetype biases applied by hand-strength bucket and action type — bluff bias, call bias,
  aggression bias, sizing bias, positional bias — plus per-infoset noise. Every opponent has known
  parameters, so labels stay exact.
- **RL agents** from diverse seeds and reward shapings, and simple rule bots, for behaviour that
  perturbed equilibria never produce.
- **Bot logs**: the Annual Computer Poker Competition released hand histories; complete
  information at showdown, machine-diverse.
- **Human data** (IRC poker database; the Pluribus-versus-humans logs) as *calibration* of the
  prior's mixture weights and biases, not as the primary training set: folded hands never reveal
  cards, so human data is censored exactly where opponent modelling matters.

## 4. Hypotheses

- **H1 (in-context inference).** A 4-layer transformer trained with the DPT recipe recovers the
  exact Bayesian posterior over a 256-opponent population: `KL(exact ‖ model) < 0.1` nats by hand
  32, and its read-out entropy tracks the exact entropy within 0.2 nats.
- **H2 (exploitation).** Its per-hand EV is within 0.02 chips/hand of the exact Bayes-myopic
  agent from hand 16 on, exceeds equilibrium from hand 4 on, and approaches oracle best response.
- **H3 (information pricing).** The value-of-information gap — oracle minus Bayes-myopic,
  cumulated — is measurable in Kuhn; IDS labels or RL fine-tuning recover part of it; the
  in-context agent's early-hand call-down and bluff frequencies differ from its late-hand ones
  against the same opponent.
- **H4 (generalisation).** Trained on the continuous prior, the agent exploits fresh opponents
  from the same family and degrades gracefully (toward equilibrium EV) on out-of-prior opponents.
- **H5 (safety).** Exploitability rises as the agent commits; a KL-to-equilibrium penalty buys a
  large exploitability reduction for a small share of exploitation EV.
- **H6 (scale).** The gains survive Kuhn → Leduc → limit hold'em → HUNL with a hierarchical
  hand→session model and a search hybrid; the opponent-parameter head remains calibrated.

## 5. Phases and experiments

| phase | game | what is exact | experiments |
|------:|------|---------------|-------------|
| 0 | Kuhn | everything: EV, BR, exploitability, posterior | **E1** in-context Bayes (spec: `docs/experiments/e1-kuhn.md`); **E2** label variants Thompson vs Bayes-BR vs IDS vs RL fine-tune, size of the value-of-information gap; **E3** continuous prior, out-of-prior opponents, ablate the opponent head |
| 1 | Leduc | EV, BR, exploitability (≈1k infosets) | **E4** perturbed-CFR population, does inference scale; **E5** safety knobs (KL to equilibrium, RNR mixing), Pareto frontier |
| 2 | limit hold'em or an HUNL abstraction | nothing; Monte Carlo + AIVAT | **E6** hierarchical hand→session model, long sessions; **E7** non-stationary and adaptive opponents, self-exploitation dynamics; **E8** RL fine-tuning against a league |
| 3 | HUNL | — | **E9** search hybrid: transformer belief → depth-limited subgame solving against the posterior-mixture opponent (Restricted Nash Response with a learned model); ACPC and human data as prior calibration |

Phase 0 runs entirely on the M4 laptop in under an hour per experiment.

## 6. Positions on the original three questions

**Are current LLMs already capable?** Not as the engine. They fail at long-horizon numeric belief
tracking, at producing calibrated mixed strategies, and at counterfactual value computation
(PokerBench 2025 put frontier models near chance postflop before fine-tuning). They are useful
at the *harness* layer — summarising a profile, choosing solver configurations, handling table
talk — the Cicero split. Transformers, however, are the right tool: DPT and Algorithm Distillation
show small transformers learn in-context posterior sampling from trajectory data.

**Is there a Hilbert space for the agent?** Yes in the sense that matters (§2.4): the belief
lives in a learned latent that is a sufficient statistic of the posterior, with an explicit
linear read-out whose entropy we can measure, and the natural metric is information-geometric.

**Data, pipeline, architecture.** Train from scratch, not from Kimi or any language base:
poker tokens share nothing with web text, and a 1M–500M parameter model on billions of synthetic
hands is cheaper and better. The prior is the dataset (§3). Pipeline: supervised DPT stage →
RL fine-tune against a league with a safety regulariser → test-time search against the posterior
opponent. Architecture: decoder-only with a custom vocabulary; hierarchical hand→session model
once sessions exceed a few thousand tokens; heads for action, value, and opponent parameters.

## 7. Architecture at scale (sketch, Phase 2+)

    hand tokens ──► per-hand encoder ──► hand embeddings ──► session transformer ──► belief latent
                                                                                        │
                       ┌─────────────────────────────────────────────────────────────────┤
                       ▼                                 ▼                               ▼
              action head (policy)              opponent-parameter head          value head
                       │                                 │
                       └──── test-time search: subgame solve vs posterior-mixture opponent ────┘

## 8. Related work and what we take from it

- Southey et al. 2005, *Bayes' Bluff* — explicit Bayesian opponent modelling in poker with
  Dirichlet priors (introduced Leduc). We amortise the inference they did by hand.
- Johanson, Zinkevich & Bowling 2007 (Restricted Nash Response); Johanson & Bowling 2009 (Data
  Biased Response) — safe exploitation at the strategy level; our Phase-1 safety knob.
- Ganzfried & Sandholm 2011, 2015 — opponent modelling in large games; provably safe exploitation.
- Bard et al. 2013 — online implicit agent modelling with portfolios; a discrete-population
  baseline.
- Ortega et al. 2019, *Meta-learning of sequential strategies*; Laskin et al. 2022 (Algorithm
  Distillation); Lee et al. 2023 (Decision-Pretrained Transformer) — in-context Bayesian RL; our
  training recipe and its theory.
- Russo & Van Roy 2014/2018 — Information-Directed Sampling; our information-pricing objective.
- DeepStack 2017, Libratus 2017, Pluribus 2019, ReBeL 2020, Student of Games 2023 — equilibrium
  search; the Phase-3 search hybrid and the safety floor.
- Burch et al. 2018 (AIVAT) — variance reduction; mandatory for Phase 2 evaluation.
- Grover et al. 2018 (policy representations); Rabinowitz et al. 2018 (ToMnet); He et al. 2016
  (DRON) — learned opponent embeddings.
- Meta FAIR 2022 (Cicero) — the LM-for-language / planner-for-strategy split.
- PokerBench 2025 — LLM poker competence baseline.

## 9. Risks and open questions

- **Prior misspecification.** A real opponent outside the prior's support gets a confidently
  wrong posterior. Mitigations: the random component, exploitability monitoring, equilibrium
  fallback when the read-out's likelihood of recent observations is low.
- **DPT labels are myopic and magnitude-blind** (§2.5). Measure the gap in E1/E2 before building
  anything on top.
- **Coverage vs deployment shift.** Posteriors are invariant to our policy; context coverage is
  not. Watch for contexts (long streaks of our own folds) that training never produced.
- **Adaptive opponents** can bait the model: feed it a fake profile then switch. Non-stationary
  priors and safety constraints are the answer, both Phase 2 work.
- **Compute at HUNL.** Exact labels vanish; best responses become approximate; sessions become
  long. The search hybrid moves most of the burden to test time.
- **Is exploitation worth it?** Against strong opponents the exploitable value is small and the
  exploitability cost real. The Pareto frontier in E5 is the honest answer.

## 10. Decision log

- 2026-09-13 — Start in Kuhn, not Leduc or HUNL: every quantity exact, so the first plots are
  about the method, not about estimator noise.
- 2026-09-13 — Hand-rolled engines rather than OpenSpiel: small, inspectable, and the in-context
  session machinery is custom anyway. Cross-check against OpenSpiel later if convenient.
- 2026-09-13 — Discrete frozen population for E1a so the exact posterior exists; continuous prior
  as E1b.
- 2026-09-13 — Train from scratch; no language-model base.
- 2026-09-13 — No Docker, no services: single-machine research code (see `README.md`).
