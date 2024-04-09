"""Functionality for parameterising a molecule using BioSimSpace."""

# pylint: disable=import-outside-toplevel, import-error

from typing import Any

import pytest

from maize.core.interface import Parameter
from maize.utilities.testing import TestRig

from ._base import _BioSimSpaceBase

__all__ = ["Parameterise"]


class Parameterise(_BioSimSpaceBase):
    """
    Parameterise a molecule using BioSimSpace.

    Notes
    -----
    Install with `mamba create -f env.yaml`.

    References
    ----------
    L. O. Hedges et al., JOSS, 2019, 4, 1831.
    L. O. Hedges et al., LiveCoMS, 2023, 5, 2375–2375.
    """

    required_callables = ["tleap"]
    """Installed with ambertools - this is a requirement for BioSimSpace"""

    # Parameters
    force_field: Parameter[str] = Parameter()
    """
    The force field with which to parameterise the molecule.
    Supported force fields are shown with BioSimSpace.Parameters.forceFields()
    e.g.:

    Protein force fields:
    - ff99
    - ff99SB
    - ff99SBildn
    - ff14SB

    Small molecule force fields:
    - gaff
    - gaff2
    - openff_unconstrained-1.0.0
    - openff_unconstrained-2.0.0
    """

    water_model: Parameter[str] = Parameter(default="tip3p")
    """
    The water model to use. Supported water models are shown with
    BioSimSpace.Solvent.waterModels() e.g.:

    - spc
    - spce
    - tip3p
    - tip4p
    - tip5p
    """

    def run(self) -> None:
        import BioSimSpace as BSS

        # Get the input
        system = self._load_input()

        # Raise an error if more than one molecule is supplied
        if len(system) > 1:
            raise ValueError("Only one molecule can be parameterised at a time.")

        # Parameterise the molecule
        param_mol = (
            BSS.Parameters.parameterise(
                system[0],
                forcefield=self.force_field.value,
                water_model=self.water_model.value,
                work_dir=str(self.work_dir),
            )
            .getMolecule()
            .toSystem()
        )

        # Save the output
        self._save_output(param_mol)


@pytest.fixture
def protein_pdb_path(shared_datadir: Any) -> Any:
    return shared_datadir / "protein.pdb"


class TestSuiteParameterise:
    def test_biosimspace_parameterisation(
        self,
        temp_working_dir: Any,
        protein_pdb_path: Any,
    ) -> None:
        """Test the BioSimSpace parameterisation node."""
        rig = TestRig(Parameterise)
        res = rig.setup_run(
            inputs={"inp": [protein_pdb_path]}, parameters={"force_field": "ff14sb"}
        )
        output = res["out"].get()
        # Get the file name from the path
        file_names = {f.name for f in output}
        assert file_names == {"bss_system.gro", "bss_system.top"}
