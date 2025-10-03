from common.config import Config
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path

from matplotlib.animation import FFMpegWriter
import logging


# ============================
# Recorder wrapper
# ============================

class Recorder:
    def __init__(self, cfg: Config, fig):
        self.cfg = cfg
        self.fig = fig
        self.writer: Optional[FFMpegWriter] = None
        self.outfile: Path = cfg.outfile()

    def __enter__(self):
        metadata = dict(artist=self.cfg.artist, comment=self.cfg.comment)
        print(self.cfg.fps)
        self.writer = FFMpegWriter(
            fps=self.cfg.fps,
            codec=self.cfg.codec,
            extra_args=[
                '-pix_fmt', 'yuv420p',
                '-loglevel', 'warning',  # keep console clean; switch to 'verbose' for debugging
            ],
            metadata=metadata,
        )
        self.writer.setup(self.fig, str(self.outfile), dpi=self.cfg.dpi)
        return self

    def grab(self):
        assert self.writer is not None
        self.writer.grab_frame()

    def __exit__(self, exc_type, exc, tb):
        if self.writer is not None:
            try:
                self.writer.finish()
            except Exception:
                pass
        logging.info("Saved: %s", self.outfile)

