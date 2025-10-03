from common.scene_artists import Scene, ArtistFactory
from common.arrow_manager import ArrowManager
from common.artists_updater import ArtistUpdater


class Visualizer:
    def __init__(self, cfg, env_adapter, model):
        self.cfg = cfg
        self.env = env_adapter
        self.scene = Scene(cfg)
        factory = ArtistFactory(self.scene.ax, cfg)
        artists = factory.create_entities(self.env.world)
        self.updater = ArtistUpdater(self.env.world, artists)
        self.arrows = None
        self.model = model

    def on_step(self):
        # 矢印データをモデルから取得
        links = self.model.links(self.env.world)  # [(i, j, w), ...]
        thr = getattr(self.cfg, "arrow_on_threshold", 0.5)
        pairs, labels = [], {}
        for (i, j, w) in links:
            if w > 0.0 and w >= thr:
                pairs.append((i, j))
                labels[(i, j)] = f"{w:.2f}"

        if self.arrows is None:
            self.arrows = ArrowManager(self.scene.ax, self.cfg, self.env.world, pairs, labels)
        else:
            self.arrows.set_graph(pairs, labels)

        self.updater.update_entities()
        if self.arrows is not None:
            self.arrows.update()

        self.scene.draw()
