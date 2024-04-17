"""Functionality for Combining BioSimSpace systems."""

# pylint: disable=import-outside-toplevel, import-error

from pathlib import Path
from typing import Any

import pytest

from maize.core.interface import Input
from maize.utilities.testing import TestRig

from ._base import _BioSimSpaceBase
from .enums import BSSEngine

__all__ = ["Combine"]


class Combine(_BioSimSpaceBase):
    """
    Combine the supplied BSS systems into a single system.

    Notes
    -----
    Install with `mamba create -f env.yaml`.

    References
    ----------
    L. O. Hedges et al., JOSS, 2019, 4, 1831.
    L. O. Hedges et al., LiveCoMS, 2023, 5, 2375–2375.
    """

    bss_engine = BSSEngine.NONE

    # Input
    inp: Input[list[Path | list[Path]]] = Input()
    """
    A list of paths to system input files. These can be in any of the formats
    given by BSS.IO.fileFormats():

    gro87, grotop, mol2, pdb, pdbx, prm7, rst rst7, psf, sdf
    """

    def run(self) -> None:
        import BioSimSpace as BSS

        # Get the input
        systems = self._load_input()

        # Combine the systems
        combined_system = systems[0]
        for sys in systems[1:]:
            combined_system += sys

        # Save the output
        self._save_output(combined_system)

    def _load_input(self) -> "BioSimSpace._SireWrappers.System":
        """Load all of the input files."""
        import BioSimSpace as BSS

        input_files_lists = self.inp.receive()

        # Convert to strings for BSS
        input_files = []
        for input_files_list in input_files_lists:
            input_files.append(
                [str(f) for f in input_files_list]
                if isinstance(input_files_list, list)
                else str(input_files_list)
            )

        # Load the input files
        systems = [BSS.IO.readMolecules(f) for f in input_files]

        return systems


class TestSuiteCombine:
    def test_biosimspace_combine_multi(
        self,
        temp_working_dir: Any,
        complex_prm7_path: Any,
        complex_rst7_path: Any,
    ) -> None:
        """
        Test the BioSimSpace combine node. We'll just combine two
        identical systems for simplicity.
        """
        rig = TestRig(Combine)
        res = rig.setup_run(
            inputs={
                "inp": [
                    [
                        [complex_prm7_path, complex_rst7_path],
                        [complex_prm7_path, complex_rst7_path],
                    ],
                ]
            }
        )
        output = res["out"].get()
        # Get the file name from the path
        file_names = {f.name for f in output}
        assert file_names == {"bss_system.gro", "bss_system.top"}

    def test_biosimspace_combine_single(
        self,
        temp_working_dir: Any,
        complex_prm7_path: Any,
        complex_rst7_path: Any,
    ) -> None:
        """Test the BioSimSpace combine node with a single system."""
        rig = TestRig(Combine)
        res = rig.setup_run(
            inputs={
                "inp": [
                    [
                        [complex_prm7_path, complex_rst7_path],
                    ],
                ]
            }
        )
        output = res["out"].get()
        # Get the file name from the path
        file_names = {f.name for f in output}
        assert file_names == {"bss_system.gro", "bss_system.top"}
