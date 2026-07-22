"""PPO training for single-intersection scenarios (cologne1).

    from ppo import train
    train()                                          # seed=42, default reward
    train(seed=43, reward_fn="pressure")

If a previous run of the same config died partway, training resumes from
the furthest checkpoint instead of starting over.
"""
import time
from pathlib import Path

import gymnasium as gym
import sumo_rl
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from utils.callbacks import TSCMetricsCallback
from utils.resume import latest_checkpoint

# Checkpoint often enough that a crash costs minutes, not hours
SAVE_FREQ = 20_000


def train(seed=42, reward_fn="diff-waiting-time",
          scenario="cologne1", timesteps=200_000):
    """Train PPO. Returns the path to the saved final model.

    - If `final.zip` already exists, does nothing.
    - If intermediate checkpoints exist, resumes from the furthest one.
    - Otherwise trains from scratch.
    """
    tag = f"seed{seed}"
    if reward_fn != "diff-waiting-time":
        tag += f"_{reward_fn}"

    tb_dir = Path("tb") / scenario / "ppo" / tag
    output_dir = Path("outputs") / scenario / "ppo" / tag
    checkpoint_dir = Path("checkpoints") / scenario / "ppo" / tag
    final_path = checkpoint_dir / "final.zip"

    if final_path.exists():
        print(f"[ppo] {scenario}/{tag}: done, skipping")
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

    # Resume if a partial run left checkpoints behind
    ckpt_path, steps_done = latest_checkpoint(checkpoint_dir, "ppo")
    if ckpt_path is not None and steps_done < timesteps:
        print(f"[ppo] {scenario}/{tag}: resuming from {steps_done:,} steps")
        model = PPO.load(str(ckpt_path), env=env, tensorboard_log=str(tb_dir))
        remaining = timesteps - steps_done
        reset_timesteps = False
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            tensorboard_log=str(tb_dir),
            seed=seed,
            verbose=0,   # quiet: watch progress in TensorBoard instead
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
        remaining = timesteps
        reset_timesteps = True

    callbacks = [
        TSCMetricsCallback(),
        CheckpointCallback(save_freq=SAVE_FREQ, save_path=str(checkpoint_dir),
                           name_prefix="ppo"),
    ]

    print(f"[ppo] {scenario}/{tag}: training {remaining:,} steps ...", flush=True)
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            tb_log_name=scenario,
            reset_num_timesteps=reset_timesteps,
            progress_bar=False,   # tqdm in a notebook over long runs bloats output
        )
        # Only mark final if learn() actually completed
        model.save(str(final_path))
        print(f"[ppo] {scenario}/{tag}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        env.close()

    return final_path
