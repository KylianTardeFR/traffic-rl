# traffic-optimization-RL

Reinforcement Learning for urban traffic signal optimization on the RESCO
benchmark scenarios (Cologne).

## Structure

```
traffic-optimization-rl/
├── experiments.ipynb       # The single notebook — run top to bottom
├── ppo.py                  # train() — single-agent PPO (cologne1)
├── dqn.py                  # train() — single-agent DQN (cologne1)
├── ippo.py                 # train() — shared-policy IPPO (cologne3, cologne8)
├── fixed_time.py           # run()   — default signal plan, any scenario
├── max_pressure.py         # run()   — Varaiya max-pressure, any scenario
├── utils/
│   ├── callbacks.py        # TSCMetricsCallback for TensorBoard
│   └── evaluate.py         # evaluate() + evaluate_marl()
├── requirements.txt
└── README.md
```

Algorithm modules expose a single function (`train` or `run`). All
orchestration lives in `experiments.ipynb`.

## What the notebook does

Three phases, run top to bottom:

1. **Baselines** — fixed-time and max-pressure on cologne1, cologne3, cologne8 (5 seeds each).
2. **Single-intersection RL** — PPO and DQN on cologne1 across 5 training seeds, plus a PPO reward-function ablation across 3 alternative rewards × 3 seeds.
3. **Multi-intersection MARL** — shared-policy IPPO on cologne3 and cologne8, 3 seeds each.

Every training cell is restartable: if `final.zip` already exists for a
(scenario, algorithm, seed, reward) tuple, training is skipped. So you can
kill the notebook anywhere and resume.

All results are written to `results/`: CSV summaries plus PNG plots.

## Setup

### 1. SUMO

**Linux / WSL2:**
```bash
sudo add-apt-repository ppa:sumo/stable -y
sudo apt update
sudo apt install sumo sumo-tools sumo-doc -y
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
source ~/.bashrc
```

**Windows native:** Download the installer from <https://sumo.dlr.de/docs/Downloads.php>.
After install, set `SUMO_HOME` to e.g. `C:\Program Files (x86)\Eclipse\Sumo`
and add `%SUMO_HOME%\bin` to PATH.

**macOS:** `brew tap dlr-ts/sumo && brew install sumo`

Verify: `sumo --version` should print ≥ 1.19.

### 2. Python environment

```bash
python -m venv .venv
# Linux/WSL/Mac:
source .venv/bin/activate
# Windows native:
.venv\Scripts\activate

pip install -r requirements.txt
```

**Note on pettingzoo:** requirements.txt pins `pettingzoo==1.24.3`. Newer
pettingzoo releases renamed `agent_selector` in a way that breaks sumo-rl
1.4.5's import (`TypeError: 'module' object is not callable`). Don't
upgrade pettingzoo independently unless you also confirm sumo-rl still
imports cleanly.

### 3. Verify

```bash
python -c "from utils.evaluate import evaluate; print('OK')"
```

### 4. Run

Open `experiments.ipynb` in Jupyter or VS Code, run all cells.

```bash
jupyter notebook experiments.ipynb
```

While training runs, monitor learning curves in a second terminal:

```bash
tensorboard --logdir tb/
```

## Linux performance on Windows — WSL2 + libsumo (10× speedup)

sumo-rl talks to SUMO via two interfaces:

- **TraCI** — socket-based, cross-platform, slow
- **libsumo** — in-process C++ binding, Linux-only, ~10× faster

`libsumo` is automatically used when available. On Windows you get TraCI by
default. The fix is to run the project inside WSL2.

### Single-run timings (cologne8, 800k timesteps, one seed)

| Platform              | Time         |
| --------------------- | ------------ |
| Windows + TraCI       | ~25 hours    |
| WSL2 + libsumo        | ~2.5 hours   |

Across the full Phase 2 + Phase 3 matrix (19+ training runs), this is the
difference between a weekend and three weeks of wall time.

### WSL2 one-time setup

**1. Install WSL2.** In PowerShell as administrator:
```powershell
wsl --install Ubuntu-22.04
```
Reboot if prompted, then complete the Ubuntu first-run setup.

**2. Inside Ubuntu** (open the WSL terminal or run `wsl` from PowerShell):
```bash
sudo apt update
sudo add-apt-repository ppa:sumo/stable -y
sudo apt install sumo sumo-tools sumo-doc python3-pip python3-venv git -y
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
source ~/.bashrc
```

**3. Clone the project in the WSL filesystem (not /mnt/c — that's slow).**
```bash
cd ~
git clone https://github.com/KylianTardeFR/traffic-optimization-RL.git
cd traffic-optimization-RL
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**4. Confirm libsumo is being used:**
```bash
python -c "import libsumo; print('libsumo OK')"
```

If that prints OK, every sumo-rl env will automatically use libsumo and
training will be ~10× faster.

### Running the notebook from WSL

**Important: `code .` must be run from inside a WSL terminal, not from
PowerShell/CMD reaching into the WSL filesystem.** If VS Code opens the
project via a `\\wsl.localhost\...` UNC path (i.e. you launched it from
Windows), the Python interpreter picker will show your `.venv` kernel as
"broken" — Windows can't execute the Linux ELF binaries inside
`.venv/bin/`.

Correct way:

1. Install the **"WSL"** extension in VS Code (publisher: Microsoft).
2. Open a WSL terminal (`wsl` from PowerShell, or the Ubuntu app).
3. `cd ~/traffic-optimization-RL && code .`
4. Confirm the bottom-left corner of VS Code shows a green
   `WSL: Ubuntu-22.04` badge — that means the window itself is running
   inside WSL, not just viewing WSL files from Windows.
5. Open `experiments.ipynb`, select the `.venv` kernel — it should now
   work rather than show "(broken)".

If a kernel is still broken after that, re-run inside the WSL terminal:
```bash
source .venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name traffic-rl --display-name "Python (traffic-rl)"
```
then reload the VS Code window and pick "Python (traffic-rl)" explicitly.

Alternative without VS Code: install Jupyter inside WSL and forward the port:
```bash
pip install jupyter
jupyter notebook --no-browser --port 8888
# Then open http://localhost:8888 in your Windows browser
```

### Caveats

- Files under `/mnt/c/...` (Windows filesystem accessed from WSL) are *very*
  slow. Keep the project under `~/` inside WSL.
- The first training run will be slower than later ones — Python / SB3 cache
  warm-up.
- `use_gui=True` won't work without an X server. Keep `use_gui=False` (the
  default everywhere in this project).

## Results layout

After running, `results/` will contain:

```
results/
├── baselines.csv               # fixed-time + max-pressure on every scenario
├── cologne1_rl.csv             # PPO + DQN + reward ablation on cologne1
├── marl.csv                    # IPPO on cologne3 / cologne8
├── final_summary.csv           # one table covering everything
├── cologne1_ppo_vs_dqn.png
├── cologne1_reward_ablation.png
└── marl_vs_baselines.png
```

Training artefacts (`checkpoints/`, `tb/`, `outputs/`) are in `.gitignore`
and stay local. Only the `results/` outputs should be pushed.

## Known gotchas already fixed in this codebase

- **`evaluate.py` / `utils/evaluate.py`**: SB3's `EvalCallback` only logs
  reward + episode length to `evaluations.npz` — sumo-rl's per-step traffic
  CSVs never get written during its eval loop. `utils/evaluate.py` does
  evaluation explicitly after training instead, capturing
  `system_mean_waiting_time` etc. directly.
- **`max_pressure.py` on multi-intersection scenarios**: `sumo_rl.parallel_env()`
  wraps the real environment through several PettingZoo wrapper layers
  before reaching the object that holds `.traffic_signals`. Naive attribute
  chasing (`env.aec_env.env.traffic_signals`) breaks because
  `SumoEnvironmentPZ` doesn't forward attribute lookups the way wrapper
  classes do. `_find_traffic_signals()` walks the chain dynamically instead
  of hardcoding a hop count, so it keeps working if sumo-rl or pettingzoo
  restructure their wrappers in a future release.
