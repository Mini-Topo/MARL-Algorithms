from common.config import Config
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ============================
# Arrow & label manager
# ============================

class ArrowManager:
    def __init__(self, ax, cfg: Config, world, 
                 pairs: Sequence[Tuple[int, int]], 
                 labels: Optional[Dict[Tuple[int, int], str]] = None):
        self.ax = ax
        self.cfg = cfg
        self.world = world
        self.pairs = list(pairs)
        self.labels_map = labels or {}
        self.arrows: Dict[Tuple[int, int], FancyArrowPatch] = {}
        self.texts: Dict[Tuple[int, int], plt.Text] = {}
        self._init_arrows()

    # --- geometry helpers ---
    @staticmethod
    def _p(agent) -> np.ndarray:
        return np.array([float(agent.state.p_pos[0]), float(agent.state.p_pos[1])], dtype=float)

    @staticmethod
    def _r(agent, default=0.05) -> float:
        return float(getattr(agent, "size", default))

    def _offset_endpoints_parallel(self, p: np.ndarray, q: np.ndarray, ra: float, rb: float) -> Tuple[np.ndarray, np.ndarray]:
        d = q - p
        n = np.linalg.norm(d)
        if n < 1e-12:
            return p.copy(), q.copy()
        u = d / n
        nperp = np.array([-u[1], u[0]])
        p2 = p + nperp * self.cfg.arrow_lateral
        q2 = q + nperp * self.cfg.arrow_lateral
        s = p2 + u * (ra + self.cfg.arrow_pad)
        t = q2 - u * (rb + self.cfg.arrow_pad)
        return s, t

    @staticmethod
    def _midpoint_and_left_normal(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = (a + b) * 0.5
        d = b - a
        n = np.hypot(d[0], d[1]) + 1e-12
        left = np.array([-d[1] / n, d[0] / n])
        return m, left

    def _make_label_for(self, key: Tuple[int, int]) -> str:
        return self.labels_map.get(key, "")
    
    # === 新規: 矢印の組とラベルをステップ毎に差し替える ===
    def set_graph(self, pairs: Sequence[Tuple[int, int]],
                  labels: Optional[Dict[Tuple[int, int], str]] = None):
        """現在の有効な矢印の組とラベルを差し替える。
        既存オブジェクトは極力再利用し、未使用は非表示にする。
        """
        new_pairs = set(pairs)
        self.pairs = list(new_pairs)
        if labels is not None:
            self.labels_map = labels
        # 1) 既存で残す/新規で作る
        for key in new_pairs:
            if key not in self.arrows:
                # 新規生成（位置は update() 内で即追従）
                arr = FancyArrowPatch((0, 0), (0, 0),
                                      arrowstyle='->', mutation_scale=14,
                                      linewidth=1.0, color='k', zorder=3,
                                      clip_on=self.cfg.clip_on)
                self.ax.add_patch(arr)
                self.arrows[key] = arr

                txt = self.ax.text(0.0, 0.0, "",
                                   ha="center", va="center", fontsize=11,
                                   color="black", zorder=4, clip_on=self.cfg.clip_on)
                self.texts[key] = txt

            # 可視化ON（位置とテキストは update() で追従）
            self.arrows[key].set_visible(True)
            self.texts[key].set_visible(True)

        # 2) 使わなくなったものを非表示化
        for key in list(self.arrows.keys()):
            if key not in new_pairs:
                self.arrows[key].set_visible(False)
                self.texts[key].set_visible(False)

    def _init_arrows(self):
        for i, j in self.pairs:
            ai, aj = self.world.agents[i], self.world.agents[j]
            p, q = self._p(ai), self._p(aj)
            s, t = self._offset_endpoints_parallel(p, q, self._r(ai), self._r(aj))
            arr = FancyArrowPatch(posA=tuple(s), posB=tuple(t),
                                  arrowstyle='->', mutation_scale=14, linewidth=1.0, color='k', zorder=3, clip_on=self.cfg.clip_on)
            self.ax.add_patch(arr)
            self.arrows[(i, j)] = arr

            # label
            m, nleft = self._midpoint_and_left_normal(s, t)
            txt = self.ax.text(float(m[0] + nleft[0] * self.cfg.label_offset),
                               float(m[1] + nleft[1] * self.cfg.label_offset),
                               self._make_label_for((i, j)),
                               ha="center", va="center", fontsize=11, color="black", zorder=4, clip_on=self.cfg.clip_on)
            self.texts[(i, j)] = txt

    def update(self):
        # update all arrows and labels
        for (i, j), arr in self.arrows.items():
            if not arr.get_visible():
                continue

            ai, aj = self.world.agents[i], self.world.agents[j]
            p, q = self._p(ai), self._p(aj)
            s, t = self._offset_endpoints_parallel(p, q, self._r(ai), self._r(aj))
            arr.set_positions(tuple(s), tuple(t))

            # visibility if too close
            close = np.linalg.norm(p - q) < (self._r(ai) + self._r(aj) + 0.03)
            arr.set_visible(not close)

            # label follows
            txt = self.texts[(i, j)]
            m, nleft = self._midpoint_and_left_normal(s, t)
            txt.set_position((float(m[0] + nleft[0] * self.cfg.label_offset),
                              float(m[1] + nleft[1] * self.cfg.label_offset)))
            txt.set_text(self._make_label_for((i, j)))
            txt.set_visible(not close)

