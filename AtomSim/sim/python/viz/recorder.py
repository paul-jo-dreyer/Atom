"""Video / GIF export from rendered RGB frames.

Two modes:
    write_video(path, frames, fps)  — one-shot: hand it all frames at once
    VideoRecorder(path, fps)        — streaming: append_data per frame, close at end

Output format inferred from extension. `.mp4` uses H.264 via ffmpeg; `.gif`
uses imageio's Pillow plugin. Both use `imageio` (with `imageio-ffmpeg` for
mp4 — it bundles ffmpeg, no system install needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio
import numpy as np


def write_video(
    path: str | Path,
    frames: Iterable[np.ndarray],
    fps: float = 60.0,
    quality: int = 8,
) -> None:
    """Write all frames in one go. `quality` only affects mp4 (1-10, higher = larger)."""
    path = Path(path)
    ext = path.suffix.lower()
    frames = list(frames)
    if not frames:
        raise ValueError("write_video: no frames provided")

    if ext == ".mp4":
        imageio.mimwrite(
            str(path), frames, fps=fps, codec="libx264", quality=quality
        )
    elif ext == ".gif":
        imageio.mimwrite(str(path), frames, fps=fps, loop=0)
    else:
        raise ValueError(f"Unsupported video format: {ext!r}. Use .mp4 or .gif.")


class VideoRecorder:
    """Streaming writer. Use as a context manager or call close() explicitly.

    Example:
        with VideoRecorder("out.mp4", fps=60) as rec:
            for scene in scenes:
                frame = renderer.render(scene)
                rec.add_frame(frame)
    """

    def __init__(
        self,
        path: str | Path,
        fps: float = 60.0,
        quality: int = 8,
    ) -> None:
        self.path = Path(path)
        ext = self.path.suffix.lower()
        if ext == ".mp4":
            self._writer = imageio.get_writer(
                str(self.path), fps=fps, codec="libx264", quality=quality
            )
        elif ext == ".gif":
            self._writer = imageio.get_writer(str(self.path), mode="I", fps=fps, loop=0)
        else:
            raise ValueError(f"Unsupported format: {ext!r}. Use .mp4 or .gif.")
        self._closed = False

    def add_frame(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("VideoRecorder.add_frame after close()")
        self._writer.append_data(frame)

    def close(self) -> None:
        if not self._closed:
            self._writer.close()
            self._closed = True

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
