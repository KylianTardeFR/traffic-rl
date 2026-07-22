"""Helper for resuming interrupted training runs.

CheckpointCallback writes files named `{prefix}_{steps}_steps.zip`.
If a run dies partway (kernel killed, laptop slept, SSH dropped), the
final model was never saved but those intermediate checkpoints survive.
`latest_checkpoint` finds the furthest one so training can pick up from
there instead of starting over.
"""
from pathlib import Path


def latest_checkpoint(checkpoint_dir: Path, prefix: str):
    """Return (path, steps_done) of the furthest checkpoint, or (None, 0).

    Files look like: checkpoints/cologne1/ppo/seed42/ppo_150000_steps.zip
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None, 0

    best_path, best_steps = None, 0
    for p in checkpoint_dir.glob(f"{prefix}_*_steps.zip"):
        try:
            # "ppo_150000_steps" -> 150000
            steps = int(p.stem.split("_")[-2])
        except (ValueError, IndexError):
            continue
        if steps > best_steps:
            best_path, best_steps = p, steps

    return best_path, best_steps
