from dataclasses import dataclass, field

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
import getpass


# ============================
# Config & utilities
# ============================

JST = timezone(timedelta(hours=9))


def jst_timestamp() -> str:
    return datetime.now(JST).strftime("%Y%m%d_%H%M%S")


@dataclass
class Config:
    # output
    fps: int = 10
    dpi: int = 150
    codec: str = "libx264"
    outdir: Path = Path(".")
    artist: str = field(default_factory=getpass.getuser)
    comment: str = "simple_tag_v3 capture"
    preview: bool = False

    # scene
    lim: float = 2.0
    clip_on: bool = False

    # env
    seed: Optional[int] = None
    num_good: int = 2
    num_adv: int = 5
    num_obs: int = 1
    max_cycles: int = 50

    # physics tweaks
    contact_force_eps: float = 1e-8
    contact_margin_eps: float = 1e-8

    # arrows
    arrow_pad: float = 0.02
    arrow_lateral: float = 0.03
    label_offset: float = 0.1

    # file name
    def outfile(self) -> Path:
        ts = jst_timestamp()
        return self.outdir / f"run_{ts}.mp4"