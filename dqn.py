"""DQN training for single-intersection scenarios (cologne1).

    from dqn import train
    train()                                          # seed=42, default reward
    train(seed=43, reward_fn="pressure")
"""
from pathlib import Path

import gymnasium as gym
import sumo_rl
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from utils.callbacks import TSCMetricsCallback


def train(seed=42, reward_fn="diff-waiting-time",
          scenario="cologne1", timesteps=200_000):
    """Train DQN. Returns the path to the saved final model.

    Skips training if a final model already exists, so the notebook
    is restartable.
    """
    tag = f"seed{seed}"
    if reward_fn != "diff-waiting-time":
        tag += f"_{reward_fn}"

    tb_dir = Path("tb") / scenario / "dqn" / tag
    output_dir = Path("outputs") / scenario / "dqn" / tag
    checkpoint_dir = Path("checkpoints") / scenario / "dqn" / tag
    final_path = checkpoint_dir / "final.zip"

    if final_path.exists():
        print(f"[dqn] {scenario}/{tag}: already trained, skipping")
        return final_path

    sumo_path = Path(sumo_rl.__file__).parent
    scenario_path = sumo_path / "nets" / "RESCO" / scenario
    net_file = scenario_path / f"{scenario}.net.xml"
    route_file = scenario_path / f"{scenario}.rou.xml"

    env = gym.make(
        "sumo-rl-v0",
        net_file=net_file,
        route_file=route_file,
        out_csv_name=str(output_dir / "train"),
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

    model = DQN(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=str(tb_dir),
        seed=seed,
        verbose=1,
        learning_rate=1.0e-4,
        buffer_size=50000,
        learning_starts=5000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        target_update_interval=500,
    )

    callbacks = [
        TSCMetricsCallback(),
        CheckpointCallback(save_freq=50000, save_path=str(checkpoint_dir), name_prefix="dqn"),
    ]

    print(f"[dqn] training {scenario}/{tag} for {timesteps:,} steps")
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            tb_log_name=scenario,
            progress_bar=True,
        )
    finally:
        model.save(str(final_path))
        env.close()
        print(f"[dqn] saved -> {final_path}")

    return final_path
