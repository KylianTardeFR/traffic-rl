"""Max-Pressure baseline (decentralized).

For each intersection at each decision step, select the green phase that
maximizes (incoming queue) − (outgoing queue) — the classical Varaiya
max-pressure rule. In multi-intersection scenarios each signal runs MP
independently.

    from max_pressure import run
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


def max_pressure_action(ts: sumo_rl.TrafficSignal) -> int:
    """Pick the green phase with the highest (in_queue − out_queue) pressure."""
    if ts.time_since_last_phase_change < ts.min_green:
        return ts.green_phase

    controlled_links = ts.sumo.trafficlight.getControlledLinks(ts.id)

    best_phase, best_pressure = ts.green_phase, -float("inf")
    for phase_idx, phase in enumerate(ts.green_phases):
        seen_in = set()
        incoming_q = 0
        for link_idx, link_group in enumerate(controlled_links):
            if link_idx < len(phase.state) and phase.state[link_idx] in "Gg":
                lane = link_group[0][0]
                if lane not in seen_in:
                    incoming_q += ts.sumo.lane.getLastStepHaltingNumber(lane)
                    seen_in.add(lane)
        outgoing_q = sum(
            ts.sumo.lane.getLastStepHaltingNumber(l) for l in ts.out_lanes
        )
        pressure = incoming_q - outgoing_q
        if pressure > best_pressure:
            best_pressure = pressure
            best_phase = phase_idx

    return best_phase


def _run_single(scenario, eval_seeds):
    net_file, route_file = _scenario_files(scenario)
    rows = []
    for seed in eval_seeds:
        print(f"  max_pressure seed {seed} on {scenario}...", flush=True)
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
            fixed_ts=False,
            sumo_seed=seed,
            add_system_info=True,
            sumo_warnings=False,
            reward_fn="queue",
        )
        env.reset()
        raw = env.unwrapped
        waits, speeds, stopped = [], [], []
        terminated = truncated = False
        while not (terminated or truncated):
            ts_id = next(iter(raw.traffic_signals))
            ts = raw.traffic_signals[ts_id]
            action = max_pressure_action(ts)
            _, _, terminated, truncated, info = env.step(action)
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
        print(f"  max_pressure seed {seed} on {scenario}...", flush=True)
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
            fixed_ts=False,
            sumo_seed=seed,
            add_system_info=True,
            sumo_warnings=False,
            reward_fn="queue",
        )
        obs, _ = env.reset()
        # PettingZoo parallel_env exposes traffic_signals via .unwrapped on env_creator,
        # but on the wrapper itself we use .aec_env or attribute lookup:
        raw = env.unwrapped if hasattr(env, "unwrapped") else env
        # sumo-rl's parallel wrapper exposes the underlying env via `env.aec_env.env`
        # Fall back to attribute discovery:
        traffic_signals = getattr(raw, "traffic_signals", None)
        if traffic_signals is None and hasattr(env, "aec_env"):
            traffic_signals = env.aec_env.env.traffic_signals

        n_agents = len(obs)
        waits, speeds, stopped = [], [], []
        done = False
        while not done:
            actions = {
                ts_id: max_pressure_action(traffic_signals[ts_id])
                for ts_id in obs.keys()
            }
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
    """Run max-pressure on `scenario`. Auto-detects single vs multi-intersection."""
    if scenario in MARL_SCENARIOS:
        return _run_marl(scenario, eval_seeds)
    return _run_single(scenario, eval_seeds)
