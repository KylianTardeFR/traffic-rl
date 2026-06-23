"""Shared-policy IPPO for multi-intersection scenarios (cologne3, cologne8).

Uses sumo_rl.parallel_env + SuperSuit to stack every intersection into a
single SB3 vec env. One PPO policy is shared across every signal.

    from ippo import train
    train(scenario="cologne3")                          # 3 intersections
    train(scenario="cologne8", timesteps=800_000)       # 8 intersections, needs more steps
"""
from pathlib import Path

import sumo_rl
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


def train(seed=42, reward_fn="diff-waiting-time",
          scenario="cologne3", timesteps=500_000):
    """Train shared-policy IPPO on a multi-intersection RESCO scenario.

    Note on `timesteps`: with N intersections, SB3 counts each env.step()
    as N timesteps. 500k on cologne3 ≈ 166k SUMO steps; on cologne8 ≈ 62k.
    For comparable depth, scale timesteps with N (e.g. 800k for cologne8).
    """
    tag = f"seed{seed}"
    if reward_fn != "diff-waiting-time":
        tag += f"_{reward_fn}"

    tb_dir = Path("tb") / scenario / "ippo" / tag
    output_dir = Path("outputs") / scenario / "ippo" / tag
    checkpoint_dir = Path("checkpoints") / scenario / "ippo" / tag
    final_path = checkpoint_dir / "final.zip"

    if final_path.exists():
        print(f"[ippo] {scenario}/{tag}: already trained, skipping")
        return final_path

    sumo_path = Path(sumo_rl.__file__).parent
    scenario_path = sumo_path / "nets" / "RESCO" / scenario
    net_file = scenario_path / f"{scenario}.net.xml"
    route_file = scenario_path / f"{scenario}.rou.xml"

    env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=str(output_dir / "train"),
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

    # Pad obs/actions (intersections differ in geometry), then stack as one vec env
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3",
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
        CheckpointCallback(save_freq=50000, save_path=str(checkpoint_dir), name_prefix="ippo"),
    ]

    print(f"[ippo] training {scenario}/{tag} for {timesteps:,} steps")
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
        print(f"[ippo] saved -> {final_path}")

    return final_path
