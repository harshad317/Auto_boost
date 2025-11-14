"""Auto Boost package."""

from .cli import AutoBoost, build_model, build_scaler, infer_problem_type, main, parse_args

__all__ = ["AutoBoost", "build_model", "build_scaler", "infer_problem_type", "main", "parse_args"]
__version__ = "0.1.4"
