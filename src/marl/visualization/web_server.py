
from flask import Flask, render_template, jsonify
import threading
import time
from pathlib import Path

# Get the absolute path to the directory containing this script
current_dir = Path(__file__).parent

app = Flask(
    __name__,
    template_folder=current_dir / "templates",
    static_folder=current_dir / "static"
)

env_state = {}

def update_state(env):
    global env_state
    while True:
        env_state = env.get_state()
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/state')
def state():
    return jsonify(env_state)

def run_web_server(env):
    thread = threading.Thread(target=update_state, args=(env,))
    thread.daemon = True
    thread.start()
    app.run(debug=True, use_reloader=False)
