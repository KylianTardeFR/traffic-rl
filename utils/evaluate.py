"""Evaluation functions for trained models.

Two functions:
  - evaluate()       — single-agent gym env (cologne1)
  - evaluate_marl()  — multi-intersection PettingZoo env (cologne3, cologne8)

Both bypass SB3's EvalCallback CSV pipeline (which doesn't trigger
sumo-rl's save_csv during eval) and capture system metrics step-by-step
in Python. Returns one DataFrame row per eval seed.
"""
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import sumo_rl
import supersuit as ss


def _scenario_files(scenario):
    sumo_path = Path(sumo_rl.__file__).parent
    scenario_path = sumo_path / "nets" / "RESCO" / scenario
    return (
        scenario_path / f"{scenario}.net.xml",
        scenario_path / f"{scenario}.rou.xml",
    )


def evaluate(model, eval_seeds=(100, 200, 300, 400, 500),
             reward_fn="diff-waiting-time", scenario="cologne1"):
    """Evaluate a single-agent model on N SUMO seeds.

    Returns DataFrame: seed, mean_wait, mean_speed, mean_stopped, p95_wait
    """
    net_file, route_file = _scenario_files(scenario)
    rows = []

    for seed in eval_seeds:
        print(f"  eval seed {seed}...", flush=True)
        env = gym.make(
            "sumo-rl-v0",
            net_file=net_file,
            route_file=route_file,
            single_agent=True,
            use_gui=False,
            num_seconds=5400,
            begin_time=25200,
            delta_time=5,
            yellow_time=2,
            min_green=5,
            max_green=60,
            sumo_seed=seed,
            reward_fn=reward_fn,
            add_system_info=True,
            sumo_warnings=False,
        )

        obs, _ = env.reset()
        waits, speeds, stopped = [], [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            if "system_mean_waiting_time" in info: waits.append(info["system_mean_waiting_time"])
            if "system_mean_speed" in info:        speeds.append(info["system_mean_speed"])
            if "system_total_stopped" in info:     stopped.append(info["system_total_stopped"])
        env.close()

        rows.append({
            "seed": seed,
            "mean_wait":    float(np.mean(waits))   if waits   else float("nan"),
            "mean_speed":   float(np.mean(speeds))  if speeds  else float("nan"),
            "mean_stopped": float(np.mean(stopped)) if stopped else float("nan"),
            "p95_wait":     float(np.percentile(waits, 95)) if waits else float("nan"),
        })

    return pd.DataFrame(rows)


def evaluate_marl(model, eval_seeds=(100, 200, 300, 400, 500),
                  reward_fn="diff-waiting-time", scenario="cologne3"):
    """Evaluate a shared-policy IPPO model on a multi-intersection scenario.

    Returns DataFrame: seed, n_agents, mean_wait, mean_speed, mean_stopped, p95_wait
    """
    net_file, route_file = _scenario_files(scenario)
    rows = []

    for seed in eval_seeds:
        print(f"  eval seed {seed}...", flush=True)
        env = sumo_rl.parallel_env(
            net_file=net_file,
            route_file=route_file,
            use_gui=False,
            num_seconds=5400,
            begin_time=25200,
            delta_time=5,
            yellow_time=2,
            min_green=5,
            max_green=60,
            sumo_seed=seed,
            reward_fn=reward_fn,
            add_system_info=True,
            sumo_warnings=False,
        )
        # Same padding the model was trained with
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)

        obs, _ = env.reset()
        n_agents = len(obs)
        waits, speeds, stopped = [], [], []
        done = False

        while not done:
            agents = list(obs.keys())
            obs_batch = np.stack([obs[a] for a in agents])
            action_batch, _ = model.predict(obs_batch, deterministic=True)
            actions = {a: int(action_batch[i]) for i, a in enumerate(agents)}
            obs, _, terminations, truncations, infos = env.step(actions)

            if infos:
                info = next(iter(infos.values()))
                if "system_mean_waiting_time" in info: waits.append(info["system_mean_waiting_time"])
                if "system_mean_speed" in info:        speeds.append(info["system_mean_speed"])
                if "system_total_stopped" in info:     stopped.append(info["system_total_stopped"])

            done = (not obs) or all(terminations.values()) or all(truncations.values())
        env.close()

        rows.append({
            "seed": seed,
            "n_agents": n_agents,
            "mean_wait":    float(np.mean(waits))   if waits   else float("nan"),
            "mean_speed":   float(np.mean(speeds))  if speeds  else float("nan"),
            "mean_stopped": float(np.mean(stopped)) if stopped else float("nan"),
            "p95_wait":     float(np.percentile(waits, 95)) if waits else float("nan"),
        })

    return pd.DataFrame(rows)
