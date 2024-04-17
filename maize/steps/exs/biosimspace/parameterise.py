"""Functionality for parameterising a molecule using BioSimSpace."""

# pylint: disable=import-outside-toplevel, import-error

from pathlib import Path
from typing import Any

import pytest

from maize.core.interface import Input, Parameter
from maize.core.node import JobResourceConfig
from maize.utilities.testing import TestRig

from ._base import _BioSimSpaceBase
from .enums import BSSEngine

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

    bss_engine = BSSEngine.TLEAP

    # Input
    inp: Input[Path] = Input(optional=True)
    """
    Path to system input file. This can be in any of
    the formats given by BSS.IO.fileFormats(), e.g.:
    
    mol2, pdb, pdbx, sdf
    """

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
        input_file = self.inp.receive()
        input_sys = BSS.IO.readMolecules(str(input_file))

        # Raise an error if more than one molecule is supplied
        if len(input_sys) > 1:
            raise ValueError("Only one molecule can be parameterised at a time.")

        # Create a python script to be run through run_command (so that we use the
        # desired scheduling system)
        param_script = (
            "import BioSimSpace as BSS\n"
            f"system = BSS.IO.readMolecules('{input_file}')\n"
            "system = system[0]\n"
            f"param_mol = BSS.Parameters.parameterise(\n"
            f"    system,\n"
            f"    forcefield='{self.force_field.value}',\n"
            f"    water_model='{self.water_model.value}',\n"
            f"    work_dir='{self.work_dir}',\n"
            ").getMolecule().toSystem()\n"
            'BSS.IO.saveMolecules("slurm_out", param_mol, ["gro87", "grotop"])\n'
        )

        # Write the script to a file
        with open("param_script.py", "w") as f:
            f.write(param_script)

        # Run the script - we want lots of CPUs
        options = JobResourceConfig(cores_per_process=18, processes_per_node=18)
        self.run_command("python param_script.py", batch_options=options)

        # Load the output
        param_mol = BSS.IO.readMolecules(["slurm_out.gro", "slurm_out.top"])

        # Save the output
        self._save_output(param_mol)

        # Dump the data
        self._dump_data()


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
        dump_dir = Path().absolute().parents[1] / "dump"
        res = rig.setup_run(
            inputs={"inp": [protein_pdb_path]},
            parameters={"force_field": "ff14sb", "dump_to": dump_dir},
        )
        output = res["out"].get()
        # Get the file name from the path
        file_names = {f.name for f in output}
        assert file_names == {"bss_system.gro", "bss_system.top"}

        # Check that the dumping worked
        # Get the most recent directory in the dump folder
        dump_output_dir = sorted(dump_dir.iterdir())[-1]
        assert (dump_output_dir / "bss_system.gro").exists()
        assert (dump_output_dir / "bss_system.top").exists()
