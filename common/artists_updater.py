from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ============================
# Artists updater
# ============================

class ArtistUpdater:
    def __init__(self, world, artists: Dict[str, List]):
        self.world = world
        self.agent_patches = artists["agent_patches"]
        self.agent_labels = artists["agent_labels"]
        self.landmark_patches = artists["landmark_patches"]

    def update_entities(self):
        for circ, ag in zip(self.agent_patches, self.world.agents):
            x, y = float(ag.state.p_pos[0]), float(ag.state.p_pos[1])
            circ.center = (x, y)
        for txt, ag in zip(self.agent_labels, self.world.agents):
            x, y = float(ag.state.p_pos[0]), float(ag.state.p_pos[1])
            txt.set_position((x, y))
        for circ, lm in zip(self.landmark_patches, self.world.landmarks):
            x, y = float(lm.state.p_pos[0]), float(lm.state.p_pos[1])
            circ.center = (x, y)
