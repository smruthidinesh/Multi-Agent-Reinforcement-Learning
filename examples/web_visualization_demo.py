
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from marl.environments.grid_world import MultiAgentGridWorld
from marl.visualization import run_web_server
import threading
import time

def main():
    env = MultiAgentGridWorld(n_agents=2, grid_size=(5, 5))

    # Run the web server in a separate thread
    server_thread = threading.Thread(target=run_web_server, args=(env,))
    server_thread.daemon = True
    server_thread.start()

    # Run the environment loop
    obs, _ = env.reset()
    done = False
    while not done:
        actions = {i: env.action_space.sample() for i in range(env.n_agents)}
        obs, _, terminated, truncated, _ = env.step(actions)
        done = all(terminated.values()) or all(truncated.values())
        time.sleep(0.5)

if __name__ == "__main__":
    main()
