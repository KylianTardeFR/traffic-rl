"""Fixed-time baseline.

Runs the default SUMO signal plan on a scenario for N seeds. Works for
single-intersection (cologne1) and multi-intersection (cologne3, cologne8)
scenarios via fixed_ts=True.

    from fixed_time import run
    df = run(scenario="cologne1")
    df = run(scenario="cologne8")
"""
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import sumo_rl


MARL_SCENARIOS = {"cologne3", "cologne8", "ingolstadt1", "ingolstadt7", "ingolstadt21"}


def _scenario_files(scenario):
    sumo_path = Path(sumo_rl.__file__).parent
    scenario_path = sumo_path / "nets" / "RESCO" / scenario
    return (
        scenario_path / f"{scenario}.net.xml",
        scenario_path / f"{scenario}.rou.xml",
    )


def _run_single(scenario, eval_seeds):
    net_file, route_file = _scenario_files(scenario)
    rows = []
    for seed in eval_seeds:
        print(f"  fixed_time seed {seed} on {scenario}...", flush=True)
        env = gym.make(
            "sumo-rl-v0",
            net_file=net_file,
            route_file=route_file,
            single_agent=True,
            use_gui=False,
            num_seconds=5400,
            begin_time=25200,
            delta_time=5,
            fixed_ts=True,
            sumo_seed=seed,
            add_system_info=True,
            sumo_warnings=False,
        )
        env.reset()
        waits, speeds, stopped = [], [], []
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(0)
            if "system_mean_waiting_time" in info: waits.append(info["system_mean_waiting_time"])
            if "system_mean_speed" in info:        speeds.append(info["system_mean_speed"])
            if "system_total_stopped" in info:     stopped.append(info["system_total_stopped"])
        env.close()
        rows.append({
            "seed": seed,
            "mean_wait":    float(np.mean(waits)) if waits else float("nan"),
            "mean_speed":   float(np.mean(speeds)) if speeds else float("nan"),
            "mean_stopped": float(np.mean(stopped)) if stopped else float("nan"),
            "p95_wait":     float(np.percentile(waits, 95)) if waits else float("nan"),
        })
    return pd.DataFrame(rows)


def _run_marl(scenario, eval_seeds):
    net_file, route_file = _scenario_files(scenario)
    rows = []
    for seed in eval_seeds:
        print(f"  fixed_time seed {seed} on {scenario}...", flush=True)
        env = sumo_rl.parallel_env(
            net_file=net_file,
            route_file=route_file,
            use_gui=False,
            num_seconds=5400,
            begin_time=25200,
            delta_time=5,
            fixed_ts=True,
            sumo_seed=seed,
            add_system_info=True,
            sumo_warnings=False,
        )
        obs, _ = env.reset()
        n_agents = len(obs)
        waits, speeds, stopped = [], [], []
        done = False
        while not done:
            # With fixed_ts the action is ignored; pass zeros
            actions = {a: 0 for a in obs.keys()}
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
            "mean_wait":    float(np.mean(waits)) if waits else float("nan"),
            "mean_speed":   float(np.mean(speeds)) if speeds else float("nan"),
            "mean_stopped": float(np.mean(stopped)) if stopped else float("nan"),
            "p95_wait":     float(np.percentile(waits, 95)) if waits else float("nan"),
        })
    return pd.DataFrame(rows)


def run(scenario="cologne1", eval_seeds=(100, 200, 300, 400, 500)):
    """Run fixed-time on `scenario`. Auto-detects single vs multi-intersection."""
    if scenario in MARL_SCENARIOS:
        return _run_marl(scenario, eval_seeds)
    return _run_single(scenario, eval_seeds)
