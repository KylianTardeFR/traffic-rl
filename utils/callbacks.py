"""TensorBoard callback for traffic metrics."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TSCMetricsCallback(BaseCallback):
    """Log sumo-rl `system_*` metrics to TensorBoard during training.

    sumo-rl populates each info dict with system_mean_waiting_time,
    system_mean_speed, etc. when add_system_info=True. We log running
    means under the `traffic/` namespace at the end of each rollout.
    """

    TRACKED = (
        "system_mean_waiting_time",
        "system_mean_speed",
        "system_total_stopped",
        "system_total_running",
        "system_mean_travel_time",
    )

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._buf = {k: [] for k in self.TRACKED}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            for k in self.TRACKED:
                if k in info:
                    self._buf[k].append(float(info[k]))
        return True

    def _on_rollout_end(self) -> None:
        for k, vals in self._buf.items():
            if vals:
                tb_key = "traffic/" + k.replace("system_", "")
                self.logger.record(tb_key, float(np.mean(vals)))
                self._buf[k] = []
