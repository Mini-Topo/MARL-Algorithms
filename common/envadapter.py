from common.config import Config
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

from mpe2 import simple_tag_v3


# ============================
# Environment adapter
# ============================

class EnvAdapter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.env = simple_tag_v3.parallel_env(
            num_good=cfg.num_good,
            num_adversaries=cfg.num_adv,
            num_obstacles=cfg.num_obs,
            max_cycles=cfg.max_cycles,
            render_mode=None,
        )
        self.reset(cfg.seed)
        # physics tweaks
        w = self.world
        w.contact_force = cfg.contact_force_eps
        w.contact_margin = cfg.contact_margin_eps

        # 行動空間の仕様をキャッシュ（名前→行動数）
        self._n_actions: Dict[str, int] = {
            a.name: self.env.action_space(a.name).n for a in self.world.agents
        }

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            try:
                return self.env.reset(seed=seed)
            except TypeError:
                return self.env.reset()
        return self.env.reset()

    @property
    def world(self):
        # PettingZoo wrappers differ
        try:
            return self.env.aec_env.unwrapped.world
        except AttributeError:
            return self.env.unwrapped.world

    @property
    def agent_names(self) -> List[str]:
        return [a.name for a in self.world.agents]
    
    def n_actions(self, agent_name: str) -> int:
        return self._n_actions[agent_name]

    def step(self, action):
        self.env.step(action)

    def close(self):
        self.env.close()

