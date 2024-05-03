"""Test the BioSimSpace ABFE CLIs."""

from pathlib import Path
from typing import Any

import pytest

from maize.graphs.exs.biosimspace.abfe import (
    abfe_multi_isomer_exposed,
    abfe_no_prep_exposed,
    abfe_with_prep_exposed,
)


@pytest.fixture
def protein_pdb_path(shared_datadir: Path) -> Path:
    return shared_datadir / "protein.pdb"


@pytest.fixture
def ligand_sdf_path(shared_datadir: Path) -> Path:
    return shared_datadir / "ligand.sdf"


@pytest.fixture
def multi_sdf_path(shared_datadir: Path) -> Path:
    return shared_datadir / "2_benzenes.sdf"


@pytest.fixture
def t4l_protein_pdb_path(shared_datadir: Path) -> Path:
    return shared_datadir / "t4l.pdb"


@pytest.fixture
def test_config() -> Path:
    return Path("devtools/test-config.toml")


@pytest.fixture
def bound_prm7_path(shared_datadir: Path) -> Path:
    return shared_datadir / "bound_sys_t4l.prm7"


@pytest.fixture
def bound_rst7_path(shared_datadir: Path) -> Path:
    return shared_datadir / "bound_sys_t4l.rst7"


@pytest.fixture
def free_prm7_path(shared_datadir: Path) -> Path:
    return shared_datadir / "free_sys_t4l.prm7"


@pytest.fixture
def free_rst7_path(shared_datadir: Path) -> Path:
    return shared_datadir / "free_sys_t4l.rst7"


def test_abfe_from_isomers(
    mocker: Any,
    multi_sdf_path: Path,
    t4l_protein_pdb_path: Path,
    test_config: Path,
) -> None:
    """Test the ABFE CLI on a system with multiple isomers in an SDF file."""
    save_name = Path("abfe_results.csv")
    mocker.patch(
        "sys.argv",
        [
            "testing",
            "--config",
            test_config.as_posix(),
            "--lig_sdfs_file",
            multi_sdf_path.as_posix(),
            "--results_file_name",
            save_name.as_posix(),
            "--ligand_force_field",
            "gaff2",
            "--protein_pdb",
            t4l_protein_pdb_path.as_posix(),
            "--protein_force_field",
            "ff14SB",
            "--abfe_timestep",
            "4",
            "--abfe_n_replicates",
            "2",
            "--abfe_runtime",
            "0.01",
            "--abfe_runtime_generate_boresch_restraint",
            "0.1",
            # Drop the runtimes where possible
            "--prep_runtime_restrained_npt",
            "0.05",
            "--prep_runtime_unrestrained_npt",
            "0.05",
            "--abfe_estimator",
            "TI",
        ],
    )
    # Run and check that all expected files have been
    # generated
    abfe_multi_isomer_exposed()
    assert save_name.exists()
    # Check that free energies are in sane range
    with open(save_name, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            assert -10 < float(line.split(",")[2]) < -1
            assert float(line.split(",")[3]) < 3
        # For two ligaands with two replicates plus header
        assert len(lines) == 5
        breakpoint()
