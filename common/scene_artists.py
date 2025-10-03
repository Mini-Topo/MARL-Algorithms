from common.config import Config
import matplotlib.pyplot as plt
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from matplotlib import patches
from matplotlib.patheffects import withStroke



# ============================
# Scene & Artists
# ============================

class Scene:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-cfg.lim, cfg.lim)
        self.ax.set_ylim(-cfg.lim, cfg.lim)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_autoscale_on(False)
        self.running = True
        self._cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._cid_close = self.fig.canvas.mpl_connect('close_event', self._on_close)

    def _on_key(self, event):
        if event.key in ('q', 'escape'):
            self.running = False

    def _on_close(self, event):
        self.running = False

    def draw(self):
        self.fig.canvas.draw()
        if self.cfg.preview:
            # purely for preview pacing; does not affect video speed
            plt.pause(1.0 / max(1, self.cfg.fps))

    def close(self):
        try:
            self.fig.canvas.mpl_disconnect(self._cid_key)
            self.fig.canvas.mpl_disconnect(self._cid_close)
        except Exception:
            pass
        plt.close(self.fig)


class ArtistFactory:
    def __init__(self, ax, cfg: Config):
        self.ax = ax
        self.cfg = cfg

    def create_entities(self, world) -> Dict[str, List]:
        agent_patches: List[patches.Circle] = []
        agent_labels: List = []
        for ag in world.agents:
            r = float(getattr(ag, "size", 0.05))
            color = "tab:red" if "adversary" in ag.name else "tab:green"
            circ = patches.Circle((float(ag.state.p_pos[0]), float(ag.state.p_pos[1])),
                                  r, facecolor=color, edgecolor='black', linewidth=0.8, zorder=2, clip_on=self.cfg.clip_on)
            self.ax.add_patch(circ)
            agent_patches.append(circ)

            # label at center
            try:
                num = int(ag.name.split('_')[-1])
            except Exception:
                num = len(agent_labels)
            t = self.ax.text(float(ag.state.p_pos[0]), float(ag.state.p_pos[1]), str(num),
                             ha="center", va="center", color="white", fontsize=12, zorder=3,
                             path_effects=[withStroke(linewidth=1.0, foreground="black")],
                             clip_on=self.cfg.clip_on)
            agent_labels.append(t)

        landmark_patches: List[patches.Circle] = []
        for lm in world.landmarks:
            r = float(getattr(lm, "size", 0.06))
            circ = patches.Circle((float(lm.state.p_pos[0]), float(lm.state.p_pos[1])),
                                  r, facecolor='0.6', edgecolor='k', linewidth=0.6, alpha=0.7, zorder=1, clip_on=self.cfg.clip_on)
            self.ax.add_patch(circ)
            landmark_patches.append(circ)

        return {
            "agent_patches": agent_patches,
            "agent_labels": agent_labels,
            "landmark_patches": landmark_patches,
        }
