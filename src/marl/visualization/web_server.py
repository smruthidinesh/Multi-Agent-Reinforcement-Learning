
from flask import Flask, render_template, jsonify
import threading
import time
import random
from pathlib import Path

# Get the absolute path to the directory containing this script
current_dir = Path(__file__).parent

app = Flask(
    __name__,
    template_folder=current_dir / "templates",
    static_folder=current_dir / "static"
)

env_state = {}
fallback_game = {
    "grid_size": [5, 5],
    "agents": [
        {"id": 0, "pos": [0, 0]},
        {"id": 1, "pos": [4, 4]},
    ],
    "targets": [
        {"id": 0, "pos": [2, 2]},
        {"id": 1, "pos": [1, 3]},
        {"id": 2, "pos": [3, 1]},
    ],
}
state_lock = threading.Lock()

def update_state(env):
    global env_state
    while True:
        env_state = env.get_state()
        time.sleep(0.1)


def _step_fallback_game():
    """Advance a lightweight simulation for serverless deployments."""
    rows, cols = fallback_game["grid_size"]

    for agent in fallback_game["agents"]:
        move = random.choice([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)])
        next_r = max(0, min(rows - 1, agent["pos"][0] + move[0]))
        next_c = max(0, min(cols - 1, agent["pos"][1] + move[1]))
        agent["pos"] = [next_r, next_c]

    occupied = {tuple(agent["pos"]) for agent in fallback_game["agents"]}
    new_targets = []
    for target in fallback_game["targets"]:
        if tuple(target["pos"]) in occupied:
            new_targets.append(
                {
                    "id": target["id"],
                    "pos": [random.randint(0, rows - 1), random.randint(0, cols - 1)],
                }
            )
        else:
            new_targets.append(target)
    fallback_game["targets"] = new_targets

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/state')
def state():
    with state_lock:
        if env_state:
            return jsonify(env_state)
        _step_fallback_game()
        return jsonify(fallback_game)

def run_web_server(env):
    thread = threading.Thread(target=update_state, args=(env,))
    thread.daemon = True
    thread.start()
    app.run(debug=True, use_reloader=False)
