"""PPO training for single-intersection scenarios (cologne1).

    from ppo import train
    train()                                          # seed=42, default reward
    train(seed=43, reward_fn="pressure")
"""
from pathlib import Path

import gymnasium as gym
import sumo_rl
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from utils.callbacks import TSCMetricsCallback


def train(seed=42, reward_fn="diff-waiting-time",
          scenario="cologne1", timesteps=200_000):
    """Train PPO. Returns the path to the saved final model.

    Skips training if a final model already exists, so the notebook
    is restartable.
    """
    tag = f"seed{seed}"
    if reward_fn != "diff-waiting-time":
        tag += f"_{reward_fn}"

    tb_dir = Path("tb") / scenario / "ppo" / tag
    output_dir = Path("outputs") / scenario / "ppo" / tag
    checkpoint_dir = Path("checkpoints") / scenario / "ppo" / tag
    final_path = checkpoint_dir / "final.zip"

    if final_path.exists():
        print(f"[ppo] {scenario}/{tag}: already trained, skipping")
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

    model = PPO(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=str(tb_dir),
        seed=seed,
        verbose=1,
        learning_rate=3.0e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    callbacks = [
        TSCMetricsCallback(),
        CheckpointCallback(save_freq=50000, save_path=str(checkpoint_dir), name_prefix="ppo"),
    ]

    print(f"[ppo] training {scenario}/{tag} for {timesteps:,} steps")
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
        print(f"[ppo] saved -> {final_path}")

    return final_path
