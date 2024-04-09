"""Functionality for solvating a molecule using BioSimSpace."""

# pylint: disable=import-outside-toplevel, import-error

from typing import Any

import pytest

from maize.core.interface import Parameter
from maize.utilities.testing import TestRig

from ._base import _BioSimSpaceBase

__all__ = ["Solvate"]


class Solvate(_BioSimSpaceBase):
    """
    Solvate a system using BioSimSpace.

    Notes
    -----
    Install with `mamba create -f env.yaml`.

    References
    ----------
    L. O. Hedges et al., JOSS, 2019, 4, 1831.
    L. O. Hedges et al., LiveCoMS, 2023, 5, 2375–2375.
    """

    required_callables = ["gmx"]
    """BioSimSpace uses gromacs for solvation"""

    # Parameters
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

    ion_conc: Parameter[float] = Parameter(default=0.15)
    """The ion concentration to use, in M of NaCl."""

    padding: Parameter[float] = Parameter(default=15)
    """
    The amount of padding, in Angstroms, to add between the molecule(s) 
    and the edge of the box. Note that any waters already present are
    ignored.
    """

    box_type: Parameter[str] = Parameter(default="rhombicDodecahedronHexagon")
    """
    The box type to use for solvation. Supported box types are shown with 
    BioSimSpace.Solvent.boxTypes(). There are:

    - cubic
    - rhombicDodecahedronHexagon
    - rhombicDodecahedronSquare
    - truncatedOctahedron
    """

    def run(self) -> None:
        import BioSimSpace as BSS

        # Map box types to BSS functions
        box_types = {
            "cubic": BSS.Box.cubic,
            "rhombicDodecahedronHexagon": BSS.Box.rhombicDodecahedronHexagon,
            "rhombicDodecahedronSquare": BSS.Box.rhombicDodecahedronSquare,
            "truncatedOctahedron": BSS.Box.truncatedOctahedron,
        }

        # Get the input
        system = self._load_input()

        # Get the "dry" system (we want to ignore any crystallographic water
        # molecules when working out the padding)
        non_waters = [mol for mol in system if mol.nAtoms() != 3]
        dry_system = BSS._SireWrappers._system.System(non_waters)
        box_min, box_max = dry_system.getAxisAlignedBoundingBox()

        # Work out the box size from the difference in the coordinates
        box_size = [y - x for x, y in zip(box_min, box_max)]

        # Add requested padding to the box size in each dimension
        padding = self.padding.value * BSS.Units.Length.angstrom

        # Work out an appropriate box. This box length will used in each dimension
        # to ensure that the cutoff constraints are satisfied if the molecule rotates
        box_length = max(box_size) + 2 * padding
        box, angles = box_types[self.box_type.value](box_length)

        # Exclude waters if they are too far from the protein. These are unlikely
        # to be important for the simulation and including them would require a larger
        # box. Exclude if further than the supplied padding.
        try:
            waters_to_exclude = [
                wat
                for wat in system.search(
                    f"water and not (water within {self.padding.value} of protein)"
                ).molecules()
            ]
            if len(waters_to_exclude) > 0:
                self.logger.info(
                    f"Excluding {len(waters_to_exclude)} waters that are over {self.padding.value} A from the protein"
                )
        except ValueError:
            waters_to_exclude = []
        system.removeMolecules(waters_to_exclude)

        self.logger.info(
            f"Solvating system with {self.water_model.value} water and {self.ion_conc.value} M NaCl..."
        )
        solvated_system = BSS.Solvent.solvate(
            model=self.water_model.value,
            molecule=system,
            box=box,
            angles=angles,
            ion_conc=self.ion_conc.value,
        )

        # Save the output
        self._save_output(solvated_system)


@pytest.fixture
def complex_prm7_path(shared_datadir: Any) -> Any:
    return shared_datadir / "complex_dry.prm7"


@pytest.fixture
def complex_rst7_path(shared_datadir: Any) -> Any:
    return shared_datadir / "complex_dry.rst7"


class TestSuiteSolvate:
    def test_biosimspace_solvation(
        self,
        temp_working_dir: Any,
        complex_prm7_path: Any,
        complex_rst7_path: Any,
    ) -> None:
        """
        Test the BioSimSpace solvation node. Note that the input is already solvated,
        but we increase the padding.
        """

        rig = TestRig(Solvate)
        res = rig.setup_run(
            inputs={"inp": [[complex_prm7_path, complex_rst7_path]]},
            parameters={"padding": 15.0, "box_type": "cubic"},
        )
        output = res["out"].get()
        # Get the file name from the path
        file_names = {f.name for f in output}
        assert file_names == {"bss_system.gro", "bss_system.top"}

        # Read the .gro file and check the box size
        with open(output[0], "r") as f:
            lines = f.readlines()
            box_size = [float(x) for x in lines[-1].split()]
            assert box_size == [7.239, 7.239, 7.239]
