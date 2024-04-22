"""Test the BioSimSpace system preparation CLIs."""

from pathlib import Path

import pytest
from typing import Any

from maize.graphs.exs.biosimspace.system_preparation import system_prep_bound_exposed, system_prep_free_exposed


@pytest.fixture
def protein_pdb_path(shared_datadir: Path) -> Path:
    return shared_datadir / "protein.pdb"


@pytest.fixture
def ligand_sdf_path(shared_datadir: Path) -> Path:
    return shared_datadir / "ligand.sdf"


@pytest.fixture
def test_config() -> Path:
    return Path("devtools/test-config.toml")


def test_system_preparation_free(
    mocker: Any,
    ligand_sdf_path: Path,
    test_config: Path,
) -> None:
    """Test preparation of a free system."""
    save_name = Path("free_prep_out")
    save_files = [save_name.with_suffix(s) for s in [".prm7", ".rst7"]]
    mocker.patch(
        "sys.argv",
        [
            "testing",
            "--config",
            test_config.as_posix(),
            "--inp",
            ligand_sdf_path.as_posix(),
            "--ligand_force_field",
            "gaff2",
            # Drop the runtimes where possible
            "--runtime_unrestrained_npt",
            "0.05",
            "--save_name",
            save_name.as_posix(),
        ],
    )
    # Run and check that all expected files have been
    # generated
    system_prep_free_exposed()
    assert all(f.exists() for f in save_files)


def test_system_preparation_bound(
    mocker: Any,
    ligand_sdf_path: Path,
    protein_pdb_path: Path,
    test_config: Path,
) -> None:
    """Test preparation of a bound system."""
    save_name = Path("bound_prep_out")
    save_files = [save_name.with_suffix(s) for s in [".prm7", ".rst7"]]
    mocker.patch(
        "sys.argv",
        [
            "testing",
            "--config",
            test_config.as_posix(),
            "--inp",
            ligand_sdf_path.as_posix(),
            "--ligand_force_field",
            "gaff2",
            "--protein_pdb",
            protein_pdb_path.as_posix(),
            "--protein_force_field",
            "ff14SB",
            # Drop the runtimes where possible
            "--runtime_restrained_npt",
            "0.05",
            "--runtime_unrestrained_npt",
            "0.05",
            "--save_name",
            save_name.as_posix(),
        ],
    )
    # Run and check that all expected files have been
    # generated
    system_prep_bound_exposed()
    assert all(f.exists() for f in save_files)
