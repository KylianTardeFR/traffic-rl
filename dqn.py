"""DQN training for single-intersection scenarios (cologne1).

    from dqn import train
    train()                                          # seed=42, default reward
    train(seed=43, reward_fn="pressure")

If a previous run of the same config died partway, training resumes from
the furthest checkpoint (including its replay buffer) instead of starting
over.
"""
import time
from pathlib import Path

import gymnasium as gym
import sumo_rl
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from utils.callbacks import TSCMetricsCallback
from utils.resume import latest_checkpoint

SAVE_FREQ = 20_000


def train(seed=42, reward_fn="diff-waiting-time",
          scenario="cologne1", timesteps=200_000):
    """Train DQN. Returns the path to the saved final model.

    - If `final.zip` already exists, does nothing.
    - If intermediate checkpoints exist, resumes from the furthest one.
    - Otherwise trains from scratch.
    """
    tag = f"seed{seed}"
    if reward_fn != "diff-waiting-time":
        tag += f"_{reward_fn}"

    tb_dir = Path("tb") / scenario / "dqn" / tag
    output_dir = Path("outputs") / scenario / "dqn" / tag
    checkpoint_dir = Path("checkpoints") / scenario / "dqn" / tag
    final_path = checkpoint_dir / "final.zip"

    if final_path.exists():
        print(f"[dqn] {scenario}/{tag}: done, skipping")
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

    ckpt_path, steps_done = latest_checkpoint(checkpoint_dir, "dqn")
    if ckpt_path is not None and steps_done < timesteps:
        print(f"[dqn] {scenario}/{tag}: resuming from {steps_done:,} steps")
        model = DQN.load(str(ckpt_path), env=env, tensorboard_log=str(tb_dir))
        # Restore the replay buffer too, otherwise DQN restarts exploration cold
        buf = checkpoint_dir / f"dqn_replay_buffer_{steps_done}_steps.pkl"
        if buf.exists():
            model.load_replay_buffer(str(buf))
            print(f"[dqn] restored replay buffer ({model.replay_buffer.size():,} transitions)")
        remaining = timesteps - steps_done
        reset_timesteps = False
    else:
        model = DQN(
            policy="MlpPolicy",
            env=env,
            tensorboard_log=str(tb_dir),
            seed=seed,
            verbose=0,   # quiet: watch progress in TensorBoard instead
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
        remaining = timesteps
        reset_timesteps = True

    callbacks = [
        TSCMetricsCallback(),
        CheckpointCallback(save_freq=SAVE_FREQ, save_path=str(checkpoint_dir),
                           name_prefix="dqn", save_replay_buffer=True),
    ]

    print(f"[dqn] {scenario}/{tag}: training {remaining:,} steps ...", flush=True)
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            tb_log_name=scenario,
            reset_num_timesteps=reset_timesteps,
            progress_bar=False,   # tqdm in a notebook over long runs bloats output
        )
        model.save(str(final_path))
        print(f"[dqn] {scenario}/{tag}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        env.close()

    return final_path
