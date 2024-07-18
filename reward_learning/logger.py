import warnings
from typing import Any

import numpy as np

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    warnings.warn("WandB not detected therefore unavailable...")
    HAS_WANDB = False


class Logger:
    def __init__(self, run_cfg: dict, sync_tensorboard: bool = True):
        assert (
            HAS_WANDB
        ), "Dependency missing. Install WandB by 'pip install wandb'"
        self._log = run_cfg.pop("log")
        if self._log:
            if run_cfg["run_id"] is None:
                self.wandb_run = wandb.init(
                    project=run_cfg["wandb_project"],
                    name=run_cfg["wandb_run_name"],
                    group=run_cfg["wandb_group"],
                    sync_tensorboard=sync_tensorboard,
                    config=run_cfg,
                )
            else:
                self.wandb_run = wandb.init(
                    id=run_cfg["run_id"],
                    project=run_cfg["wandb_project"],
                    name=run_cfg["wandb_run_name"],
                    group=run_cfg["wandb_group"],
                    sync_tensorboard=sync_tensorboard,
                    config=run_cfg,
                    resume="must",
                )

    @property
    def to_log(self):
        return self._log

    def log(self, metrics, step):
        if self._log:
            log = {"global_step": step}
            for k, v in metrics.items():
                if (
                    isinstance(v, int)
                    or isinstance(v, float)
                    or (isinstance(v, np.ndarray) and len(v.shape) == 1)
                ):
                    log[k] = float(v)
                elif isinstance(v, np.ndarray) and (len(v.shape) in [2, 3]):
                    log[k] = wandb.Image(v, caption=k)
                elif isinstance(v, np.ndarray) and (len(v.shape) == 4):
                    log[k] = wandb.Video(v, fps=4)
                else:
                    warnings.warn(f"Unsupported log variable type for {k}")
            wandb.log(log)
