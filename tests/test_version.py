from importlib import import_module


def test_version_is_pep440_compliant():
    pkg = import_module("auto_boost")
    version = pkg.__version__
    parts = version.split(".")
    assert len(parts) == 3, "Version should follow semantic MAJOR.MINOR.PATCH format"
    assert all(part.isdigit() for part in parts), "Version segments must be numeric"
