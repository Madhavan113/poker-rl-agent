import importlib.metadata as md
import pyspiel
from open_spiel.python.algorithms import cfr, exploitability, expected_game_score

try:
    print("open_spiel version", md.version("open_spiel"))
except Exception as exc:  # noqa: BLE001
    print("version lookup failed", exc)
game = pyspiel.load_game("leduc_poker")
print("game", game, "params", game.get_parameters())
seen = set()
n_term = 0
def dfs(s):
    global n_term
    if s.is_terminal():
        n_term += 1
        return
    if s.is_chance_node():
        for a, _ in s.chance_outcomes():
            dfs(s.child(a))
        return
    seen.add((s.current_player(), s.information_state_string()))
    for a in s.legal_actions():
        dfs(s.child(a))
dfs(game.new_initial_state())
print("infosets", len(seen), "terminals", n_term, "max_utility", game.max_utility())
solver = cfr.CFRPlusSolver(game)
for i in range(1000):
    solver.evaluate_and_update_policy()
avg = solver.average_policy()
print("value_after_1000_cfrplus", expected_game_score.policy_value(game.new_initial_state(), [avg, avg]))
print("exploitability_after_1000", exploitability.exploitability(game, avg))
for i in range(4000):
    solver.evaluate_and_update_policy()
avg = solver.average_policy()
print("value_after_5000_cfrplus", expected_game_score.policy_value(game.new_initial_state(), [avg, avg]))
print("exploitability_after_5000", exploitability.exploitability(game, avg))
