# E1 — In-context Bayesian exploitation in Kuhn poker

Status: spec (2026-09-13). Companion to `RESEARCH.md` (read that first for the why).

## Goal

Show, in a game where everything is exactly computable, that a small transformer trained with
the Decision-Pretrained-Transformer (DPT) recipe on sessions sampled from an opponent prior:

1. recovers the exact Bayesian posterior over opponents in context (H1), and
2. exploits opponents nearly as well as the exact-Bayes agents, approaching oracle best response
   as hands accumulate, while dominating the equilibrium baseline (H2).

Deliverables: `runs/e1/ev_vs_hand.png`, `runs/e1/entropy_vs_hand.png`,
`runs/e1/exploitability_vs_hand.png`, `runs/e1/summary.md`.

Everything in Kuhn is exact: expected values, best responses, exploitability and the posterior are
computed by game-tree traversal, never by Monte Carlo. Realized chip results are logged only as a
sanity check.

## Game: Kuhn poker

- Cards J=0, Q=1, K=2. Each player antes 1. One card each, no board.
- Seat 0 acts first: check (CALL) or bet 1 (RAISE).
- After check: seat 1 checks (showdown, winner +1) or bets. After check-bet: seat 0 calls
  (showdown, winner +2) or folds (seat 1 wins +1).
- After bet: seat 1 calls (showdown, winner +2) or folds (seat 0 wins +1).
- Game value for seat 0 under equilibrium: −1/18.
- Nash family (α ∈ [0, 1/3]). Seat 0: bet J = α, bet Q = 0, bet K = 3α; after check-bet call
  J = 0, call Q = α + 1/3, call K = 1. Seat 1: after check bet J = 1/3, bet Q = 0, bet K = 1;
  facing bet call J = 0, call Q = 1/3, call K = 1.

Seats alternate every hand (agent is seat 0 on even hand indices). An opponent is a full
behavioural profile covering both seats: 12 infosets, 12 free parameters.

## Interfaces (shared contract for all agents)

Actions are `int`: `FOLD=0`, `CALL=1` (check or call), `RAISE=2` (bet or raise). Every strategy
vector has length 3 with zeros on illegal actions.

```python
# exsolver/games/base.py
Action = int
FOLD, CALL, RAISE = 0, 1, 2

@dataclass(frozen=True)
class GameSpec:
    name: str            # "kuhn" | "leduc"
    n_cards: int         # deck size; card indices 0..n_cards-1
    n_rounds: int
    max_result: int      # largest |chips| one player can win in a hand (Kuhn 2, Leduc 13)
    ante: int = 1

class Game(Protocol):
    spec: GameSpec
    def root(self) -> State                              # chance node, nothing dealt
    def is_chance(self, s: State) -> bool
    def chance_outcomes(self, s: State) -> list[tuple[State, float]]
    def is_terminal(self, s: State) -> bool
    def returns(self, s: State) -> tuple[float, float]   # chips won by seat 0, seat 1; zero-sum
    def current_player(self, s: State) -> int            # 0 or 1 (only at decision nodes)
    def legal_actions(self, s: State) -> list[Action]
    def apply(self, s: State, a: Action) -> State
    def infoset_key(self, s: State, player: int) -> str  # see key format below
    def private_cards(self, s: State, player: int) -> tuple[int, ...]
    def board_cards(self, s: State) -> tuple[int, ...]
    def card_name(self, c: int) -> str
    def deal(self, rng, seat_cards: ... ) -> State       # convenience: resolve chance with rng
    def state_from_deal(self, cards0, cards1, board=()) -> State  # deterministic deal for posterior replay

# exsolver/games/kuhn.py, exsolver/games/leduc.py
class KuhnPoker:    # implements Game; spec.name == "kuhn"
class LeducPoker:   # implements Game; spec.name == "leduc"
```

`State` is immutable (frozen dataclass or tuple). Infoset keys are `"{seat}:{cards}|{history}"`
where `history` uses `c` for CALL, `b` for RAISE, `f` for FOLD. Kuhn: `"0:K|"`, `"0:Q|cb"`,
`"1:J|c"`, `"1:K|b"`. Leduc, once the board is dealt, is
`"{seat}:{card}|{round1}/{board}|{round2}"`, e.g. `"1:Qs|cbc/Kh|c"` (seat 1 holding Q♠; round 1
went check, bet, call; board K♥; seat 0 checked to open round 2); during round 1 the key is just
`"{seat}:{card}|{round1}"`, e.g. `"0:Kh|b"`. Card names are rank then suit; card index =
rank·2 + suit with J, Q, K = 0, 1, 2 and s, h = 0, 1: Js=0, Jh=1, Qs=2, Qh=3, Ks=4, Kh=5.

```python
# exsolver/strategy.py
TabularStrategy = dict[str, np.ndarray]   # infoset key -> probs over (FOLD, CALL, RAISE)
def enumerate_infosets(game) -> dict[str, list[Action]]
    # every infoset key of both seats -> legal actions there (Kuhn 12 keys, Leduc 936)

# exsolver/solvers/cfr.py
def cfr_plus(game, iterations: int) -> TabularStrategy           # average strategy, both seats

# exsolver/solvers/best_response.py
def best_response(game, opponent: TabularStrategy, br_seat: int) -> tuple[TabularStrategy, float]
    # exact BR for br_seat vs opponent's other-seat entries; returns (pure-ish BR, EV for br_seat)
def best_response_value(game, opponent: TabularStrategy, br_seat: int) -> float
    # the EV component of best_response
def expected_value(game, strat_seat0: TabularStrategy, strat_seat1: TabularStrategy) -> float
    # exact EV for seat 0
def exploitability(game, profile: TabularStrategy) -> float
    # (BR value vs seat0 + BR value vs seat1) / 2, i.e. NashConv/2 ; 0 at equilibrium
def seat_exploitability(game, strat: TabularStrategy, seat: int, game_value_seat0: float) -> float
    # how much an omniscient opponent gains over their equilibrium value against `strat` at `seat`

# exsolver/solvers/mixture.py
def mix_strategies(game, weights: np.ndarray, profiles: list[TabularStrategy], seat: int) -> TabularStrategy
    # realization-equivalent behavioural mixture (sequence-form average) of profiles' `seat` entries
```

`mix_strategies` must satisfy `expected_value(σ, mix) == Σ w_i expected_value(σ, θ_i)` for any σ.

```python
# exsolver/population/kuhn_prior.py  (the prior; see RESEARCH.md "The prior")
@dataclass
class KuhnPrior:                   # generative prior over 12-dim θ; archetype mixture with Beta noise
    def sample(self, rng) -> tuple[TabularStrategy, np.ndarray, int]   # (profile, theta[12], archetype_id)
def theta_to_profile(theta: np.ndarray) -> TabularStrategy
def profile_to_theta(profile: TabularStrategy) -> np.ndarray
def nash_theta(alpha: float) -> np.ndarray   # θ of the Nash-family member with parameter α ∈ [0, 1/3]
KUHN_PARAM_NAMES: list[str]        # 12 names, fixed order:
  # ["0:J|", "0:Q|", "0:K|",        P(RAISE) at seat-0 root
  #  "0:J|cb", "0:Q|cb", "0:K|cb",  P(CALL) facing bet after checking
  #  "1:J|c", "1:Q|c", "1:K|c",     P(RAISE) after opponent checks
  #  "1:J|b", "1:Q|b", "1:K|b"]     P(CALL) facing bet

# exsolver/population/population.py
@dataclass
class Population:                  # discrete population = the prior for E1a
    profiles: list[TabularStrategy]; thetas: np.ndarray [M,12]; archetypes: np.ndarray [M]; weights: np.ndarray [M]
    def save(path) / load(path)    # .npz
def sample_population(prior: KuhnPrior, m: int, rng: np.random.Generator) -> Population
    # M draws from the prior (E1a: M = 256, seed 0)
```

```python
# exsolver/data/records.py
Event = tuple  # ("act", actor: int, action: int) with actor 0 = agent, 1 = opponent | ("board", card: int)
@dataclass
class HandRecord:
    seat: int                        # agent's seat this hand
    my_cards: tuple[int, ...]
    events: list[Event]
    opp_cards: tuple[int, ...] | None   # revealed at showdown, else None
    result: int                      # chips won by the agent (negative when losing)

@dataclass
class SessionContext:
    hands: list[HandRecord]          # completed hands, oldest first
```

```python
# exsolver/agents/base.py
class Agent(Protocol):
    name: str
    def reset(self, rng) -> None                     # start of a session
    def observe(self, hand: HandRecord) -> None      # called by the evaluator after every completed hand
    def strategy_for_hand(self, ctx: SessionContext, seat: int) -> TabularStrategy
        # full strategy over ALL of the agent's infosets at `seat` for the upcoming hand,
        # conditioned on ctx. Used both to sample the agent's actions and for exact EV/exploitability.
    def belief(self, ctx: SessionContext) -> np.ndarray | None
        # posterior over the discrete population if the agent has one (exact-Bayes agents and the
        # transformer's opponent head); None otherwise
```

## Tokenizer (game-parametrised by `GameSpec`)

Vocabulary, in this order: `PAD, BOS, HAND, END_HAND, POS_0, POS_1, CARD_0..CARD_{n-1},
BOARD_0..BOARD_{n-1}, ME_FOLD, ME_CALL, ME_RAISE, OPP_FOLD, OPP_CALL, OPP_RAISE,
SHOW_0..SHOW_{n-1}, NO_SHOW, RESULT_{-max_result}..RESULT_{+max_result}`.

A session encodes as `BOS` then per hand
`HAND POS_s CARD_c [events...] (SHOW_c | NO_SHOW) RESULT_r END_HAND`, right-padded with `PAD` to
length `L`. A Kuhn hand is at most 9 tokens; with 64 hands per session `L = 640`.

Decision positions are the positions `p` whose next token is `ME_*`; the model's hidden state at
`p` predicts the action. Belief read-out positions are `BOS` and every `END_HAND`.

## Data (DPT recipe), `exsolver/data/generate.py`

Per session: sample `opp_id ~ weights`, `θ = population[opp_id]`; get (cached) `BR(θ)` for both
seats; sample a collection policy for the whole session from
`{equilibrium: 0.4, uniform-random-over-legal: 0.3, oracle BR(θ): 0.3}`; play `H` hands with
alternating seats, agent actions from the collection policy, opponent actions from θ; record
`HandRecord`s. At every agent decision store the label `BR(θ)` action at that infoset (the *taken*
action is still whatever the collection policy did; it enters the token stream as context only).

Why any collection policy works: the posterior over θ given the observed context depends on the
agent's own actions only through θ-independent factors, so it is invariant to the collection
policy. The mixture is there for coverage of contexts the deployed agent will create itself.

Shard format (`.npz`, right-padded to `L`):

| array           | dtype   | shape        | meaning                                            |
|-----------------|---------|--------------|----------------------------------------------------|
| `tokens`        | int16   | `[N, L]`     | token ids                                          |
| `action_mask`   | bool    | `[N, L]`     | decision positions                                 |
| `action_target` | int8    | `[N, L]`     | BR action at decision positions, −1 elsewhere      |
| `legal`         | bool    | `[N, L, 3]`  | legal actions at decision positions                |
| `belief_mask`   | bool    | `[N, L]`     | BOS and END_HAND positions                         |
| `opp_id`        | int32   | `[N]`        | index into the population (−1 for continuous prior)|
| `theta`         | float32 | `[N, P]`     | opponent parameters (P = 12 for Kuhn)              |
| `archetype`     | int32   | `[N]`        | archetype id                                       |

plus `meta.json` next to the shards: game, vocab size, `L`, `H`, population path/hash, collection
mix, seed.

## Model, `exsolver/model/`

Decoder-only transformer, causal, learned positional embeddings. Defaults `d_model=128`,
`n_layers=4`, `n_heads=4`, `max_len=1024`, no dropout. Heads on the final hidden state:

- action head → 3 logits (illegal actions masked to −inf using `legal` before softmax / CE)
- opponent head → `M` logits (E1a, discrete population), trained with CE at belief positions
- theta head → `P` sigmoids (E1b, continuous prior), trained with BCE at belief positions

Loss `= CE_action + λ_opp · CE_opp + λ_theta · BCE_theta`, defaults `λ_opp = 0.5`,
`λ_theta = 0.5`. AdamW, lr 3e-4, weight decay 0.01, cosine decay, warm-up 500 steps, batch 64,
(E1 runs use batch 32 × 12 000 steps, see the configuration table; further implementation details are
declared in the `TrainConfig` docstring: gradient clipping 1.0, cosine floor at 10 % of lr, AdamW betas
(0.9, 0.95), weight decay only on ≥2-D tensors, loss-exact `trim_padding`, deterministic held-out split.)
20k steps. Device: `mps` if available else `cpu`. Checkpoint stores config, vocab meta and weights.

Inference helper: `policy_at(prefixes) -> [B, 3]` batched action distributions at the end of each
prefix, and `belief_at(prefixes) -> [B, M]`. The transformer agent implements
`strategy_for_hand` by querying every agent infoset of the upcoming hand (card × history) as a
(the unpadded session stream is appended as one more row so the belief read-out shares the forward pass)
counterfactual prefix `ctx + HAND POS_s CARD_c [history...]` — 6 queries per hand in Kuhn.

## Baseline agents, `exsolver/agents/`

| agent            | strategy for hand t                                                            |
|------------------|--------------------------------------------------------------------------------|
| `Equilibrium`    | CFR+ average strategy, fixed                                                   |
| `OracleBR`       | `BR(θ_true)`; upper bound                                                       |
| `Thompson`       | sample `θ' ~ posterior_t` (exact, discrete population) and play `BR(θ')`         |
| `BayesBR`        | `BR(mix_strategies(posterior_t, population))`; the myopic Bayes-optimal policy  |
| `Transformer`    | sampled or argmax from the action head (report both)                            |
| `Random`         | uniform over legal actions (data collection only)                               |

Exact posterior (`exsolver/bayes/posterior.py`): for each opponent `i`, the likelihood of a hand
marginalises the opponent's *unknown* private card jointly over all of that opponent's actions in
the hand (they share the same card), weighting each candidate card by the deal probability given
the agent's card and the board; if a showdown revealed the card, use it directly. Implement with
game-tree replay from `state_from_deal` and a dense `[M, n_infosets, 3]` array for speed. Keep
log-probabilities. `entropy()` in nats.

## Session runner and metrics, `exsolver/eval/`

`run_session(game, agent, opp_profile, H, rng, *, posterior=None, equilibrium_value_seat0, ...)`
(implemented signature; the evaluator owns the exact posterior object and updates it, the agent only
receives `observe(hand)`; `rng` is split into deal / opponent / agent / reset streams so every agent
faces identical deals for the same `(eval_seed, opp_id, session)`) plays `H` hands, alternating
seats, sampling the agent's actions from `strategy_for_hand` and the opponent's from θ. Per hand
index `t` it records:

- `ev[t]`: exact EV of the agent's hand-`t` strategy versus θ at that seat
- `realized[t]`: chips actually won (sanity only)
- `expl[t]`: `seat_exploitability` of the agent's hand-`t` strategy
- `post_entropy[t]`: exact posterior entropy before hand `t` (discrete population)
- `agent_entropy[t]`, `kl[t]`: entropy of `agent.belief` and `KL(exact ‖ agent.belief)`, when available
- `probe[t]`: P(CALL) facing a bet holding Q, and P(RAISE) holding J at the root — bluff and call-down
  frequencies used to look at information seeking

Aggregate over opponents × sessions; report means with standard errors. Also cumulative regret
versus `OracleBR`.

## Experiment configuration (E1a, discrete population)

| item                         | default                                       |
|------------------------------|-----------------------------------------------|
| population size `M`          | 256, sampled once from `KuhnPrior`, seed 0    |
| population weights           | uniform                                       |
| hands per session `H`        | 64                                            |
| training sessions            | 100 000                                       |
| eval                         | every opponent × 4 fresh sessions, seed 1     |
| model                        | 4 × 128, 4 heads                              |
| steps / batch                | 12 000 / 32 (spec target 20 000 / 64; see note)|

Training-budget note: on the M4/MPS the attention backward is memory-bound, so throughput is roughly
100–135 session-passes per second regardless of batch size; 20 000 × 64 would take ~3 h. The runner
therefore defaults to 32 × 12 000 (~1 h) and records the deviation in `<out>/README.md` and
`summary.md`; `--batch/--steps` override it.

E1b (continuous prior): fresh θ per session from `KuhnPrior`; theta head instead of opponent head;
eval on 512 fresh θ. No exact posterior over a discrete set; report EV curves, theta-head BCE
and calibration versus hands seen.

Runner: `uv run python -m exsolver.experiments.e1_kuhn {population,gen,train,eval,all} --out runs/e1
[--smoke] [--max-train-minutes M] [--retrain]` where `--smoke` shrinks everything to run in under two
minutes on CPU and writes to `runs/e1_smoke` (it refuses `--out runs/e1` and the real dataset directory);
every stage appends its command line and resolved config to `<out>/README.md`, and `eval` copies the
checkpoint's training provenance into `summary.md`/`summary.json`.

## Success criteria

Criteria are evaluated pointwise on the per-hand means for every `t` at or beyond the
threshold (stricter than an average by `t`); EV comparisons are paired per `(opp, session)`.

- H1: `kl[t]` for the transformer falls below 0.1 nats by `t = 32` on average, and
  `agent_entropy` tracks `post_entropy` within 0.2 nats.
- H2: transformer (sampled) mean `ev[t]` is within 0.02 chips/hand of `BayesBR` for `t ≥ 16` and
  exceeds `Equilibrium` for all `t ≥ 4`.
- Sanity: `OracleBR` ≥ every agent at every `t`; `Equilibrium` EV against the equilibrium opponent
  equals ∓1/18 by seat; `expl` of `Equilibrium` is ~0.

## Out of scope for E1

Leduc runs (engine only, Phase 1), RL fine-tuning, safety constraints, non-stationary opponents,
IDS labels. Those are E2+ in `RESEARCH.md`.

## Declared deviations (kept in sync with the merged code)

| item | spec said | implemented | where declared |
|------|-----------|-------------|----------------|
| Leduc infoset key | `"1:Qs\|cb/Kh\|c"` | `"{seat}:{card}\|{round1}/{board}\|{round2}"`, e.g. `"1:Qs\|cbc/Kh\|c"` | #2 |
| best response at zero-reach infosets | lowest-index tie rule | CALL when legal, else lowest legal (values unchanged) | #2, `solvers/best_response.py` |
| `Game` protocol | — | gains `sample_chance(s, rng)` | #2 |
| optimiser | AdamW, cosine | + grad clip 1.0, floor 10 %, betas (0.9, 0.95), decay on ≥2-D only | #3, `TrainConfig` docstring |
| `Policy.policy_at` | — | `legal` required; raises on all-False rows | #3 |
| training budget | 20 000 × 64 | 12 000 × 32 (MPS throughput) | runner README / summary |
| `Agent` protocol | 4 methods | + `observe(hand)` | wave-2 PR, `agents/base.py` |
| `run_session` | `(game, agent, theta_profile, opp_id, population, H, rng)` | see §Session runner | wave-2 PR, `eval/session.py` |
| success criteria | "by t = 32 on average" | pointwise for all t ≥ threshold | wave-2 PR, `eval/aggregate.py` |
| `--smoke` output | — | `runs/e1_smoke`, never `runs/e1` or `data/e1a` | wave-2 PR, `experiments/e1_kuhn.py` |
