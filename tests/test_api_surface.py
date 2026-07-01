"""Static API checks for the released ProEHN package."""

from pathlib import Path

import pytest


def test_release_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").exists()
    assert (root / "configs" / "default.yaml").exists()
    assert (root / "docs" / "METHODS.md").exists()


def test_public_api_names() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    from proehn import ProEHNEngine, ProEHNTopologyModel

    assert ProEHNEngine.__name__ == "ProEHNEngine"
    assert ProEHNTopologyModel.__name__ == "ProEHNTopologyModel"


def test_topology_parameter_shape() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    from proehn import ProEHNTopologyModel

    model = ProEHNTopologyModel(n_events=3, n_features=2)
    assert model.n_total == 4
    assert model.n_features == 3
    assert model.shapes.total_size == 3 * 4 * 4 + 2 * 3 * 4
