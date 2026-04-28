"""AtomSim visualization package.

Architecture:
    scene.py     — pure data: SceneSpec describing what's in the scene
    style.py     — pure data: StyleConfig describing how to draw it
    renderers/   — implementations: pygame live, pygame headless, ...

The same SceneSpec can be drawn live (interactive teleop) or headlessly
(offline video export from a recorded episode). Renderers are
interchangeable; SceneSpec and StyleConfig are the stable contract.
"""

from .episode import Episode, EpisodeRecorder
from .recorder import VideoRecorder, write_video
from .scene import (
    BallSpec,
    FieldSpec,
    RobotSpec,
    SceneSpec,
    build_scene,
)
from .style import (
    BallStyle,
    FieldStyle,
    Resolution,
    RobotStyle,
    StyleConfig,
    TeamStyle,
    load_style,
)

__all__ = [
    "BallSpec",
    "BallStyle",
    "Episode",
    "EpisodeRecorder",
    "FieldSpec",
    "FieldStyle",
    "Resolution",
    "RobotSpec",
    "RobotStyle",
    "SceneSpec",
    "StyleConfig",
    "TeamStyle",
    "VideoRecorder",
    "build_scene",
    "load_style",
    "write_video",
]
