
const gridContainer = document.getElementById('grid-container');

function drawGrid(state) {
    gridContainer.innerHTML = '';
    gridContainer.style.gridTemplateColumns = `repeat(${state.grid_size[1]}, 50px)`;
    gridContainer.style.gridTemplateRows = `repeat(${state.grid_size[0]}, 50px)`;

    for (let r = 0; r < state.grid_size[0]; r++) {
        for (let c = 0; c < state.grid_size[1]; c++) {
            const cell = document.createElement('div');
            cell.classList.add('grid-cell');

            // Check for agents
            for (const agent of state.agents) {
                if (agent.pos[0] === r && agent.pos[1] === c) {
                    cell.classList.add('agent');
                }
            }

            // Check for targets
            for (const target of state.targets) {
                if (target.pos[0] === r && target.pos[1] === c) {
                    cell.classList.add('target');
                }
            }

            gridContainer.appendChild(cell);
        }
    }
}

async function fetchState() {
    const response = await fetch('/state');
    const state = await response.json();
    drawGrid(state);
}

setInterval(fetchState, 200);
