"""Qualcomm modules, with pinned upstream modules available for FLARE imports."""
from pathlib import Path

# Direct script execution also exposes this folder as envs/utils. Preserve
# upstream absolute imports such as envs.scenario1_env and utils.mixer.
__path__.append(str(Path(__file__).resolve().parents[2] / "Flare" / "envs"))
