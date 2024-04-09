"""Functionality for minimising a system using BioSimSpace"""

# pylint: disable=import-outside-toplevel, import-error

import shutil
from abc import ABC
from typing import Any

import pytest

from maize.core.interface import Parameter
from maize.utilities.testing import TestRig

from ._base import _BioSimSpaceBase
from ._utils import _ClassProperty
from .enums import _ENGINE_CALLABLES, BSSEngine
from .exceptions import BioSimSpaceNullSystemError

__all__ = [f"Minimise{engine.class_name}" for engine in BSSEngine]


class _MinimiseBase(_BioSimSpaceBase, ABC):
    """
    Abstract base class for BioSimSpace minimisation nodes.
    All subclasses should set the biosimspace_engine attribute,
    and the required_callables attribute will be set automatically.

    Notes
    -----
    Install with `mamba create -f env.yaml`.

    References
    ----------
    L. O. Hedges et al., JOSS, 2019, 4, 1831.
    L. O. Hedges et al., LiveCoMS, 2023, 5, 2375–2375.
    """

    bss_engine: BSSEngine
    """The engine to use in the subclass."""

    @_ClassProperty
    def required_callables(cls) -> list[str]:
        return _ENGINE_CALLABLES[cls.bss_engine]

    # Parameters
    steps: Parameter[int] = Parameter(default=10_000)
    """The maximum number of steps to perform."""

    restraint: Parameter[str | list[int]] = Parameter(default=[])
    """
    The type of restraint to perform. This should be one of the
    following options:
        "backbone"
             Protein backbone atoms. The matching is done by a name
             template, so is unreliable on conversion between
             molecular file formats.
        "heavy"
             All non-hydrogen atoms that aren't part of water
             molecules or free ions.
        "all"
             All atoms that aren't part of water molecules or free
             ions.
    Alternatively, the user can pass a list of atom indices for
    more fine-grained control. If None, then no restraints are used.
    """

    force_constant: Parameter[float] = Parameter(default=10.0)
    """
    The force constant for the restraint potential, in 
    kcal_per_mol / angstrom**2
    """

    def run(self) -> None:
        import BioSimSpace as BSS

        # Map the BSS Engines to process classes
        process_map = {
            BSSEngine.GROMACS: BSS.Process.Gromacs,
            BSSEngine.SANDER: BSS.Process.Amber,
            BSSEngine.PMEMD: BSS.Process.Amber,
            BSSEngine.PMEMD_CUDA: BSS.Process.Amber,
            # BSSEngine.OPENMM: BSS.Process.OpenMM,
            BSSEngine.SOMD: BSS.Process.Somd,
            # BSSEngine.NAMD: BSS.Process.Namd,
        }

        # Get the input
        system = self._load_input()

        # Create the protocol
        protocol = BSS.Protocol.Minimisation(
            steps=self.steps.value,
            restraint=self.restraint.value if self.restraint.value else None,
            force_constant=self.force_constant.value,
        )

        # Get the process
        process = process_map[self.bss_engine](
            system,
            protocol=protocol,
            exe=self._get_executable(),
            work_dir=str(self.work_dir),
        )

        # Run the process and wait for it to finish
        self.logger.info(f"Minimising system with {self.bss_engine}...")
        cmd = " ".join([process.exe(), process.getArgString()])
        self.run_command(cmd)
        minimised_system = process.getSystem(block=True)
        # BioSimSpace sometimes returns None, so we need to check
        if minimised_system is None:
            raise BioSimSpaceNullSystemError("The minimised system is None.")

        # Save the output
        self._save_output(minimised_system)

    def _get_executable(self) -> str:
        """Get the full path to the executable."""
        if self.required_callables:
            return shutil.which(self.required_callables[0])
        return None


# Programmatically derive the Minimise classes, ensuring that we include custom docstrings
for engine in BSSEngine:
    docstring = f"""
    Minimise the system using {engine.name.capitalize()} through BioSimSpace.

    Notes
    -----
    Install with `mamba create -f env.yaml`.

    References
    ----------
    L. O. Hedges et al., JOSS, 2019, 4, 1831.
    L. O. Hedges et al., LiveCoMS, 2023, 5, 2375–2375.
    """
    # Split the engine name by underscores and capitalise each word
    class_name = f"Minimise{engine.class_name}"
    globals()[class_name] = type(
        class_name,
        (_MinimiseBase,),
        {"bss_engine": engine, "__doc__": docstring},
    )


@pytest.fixture
def complex_prm7_path(shared_datadir: Any) -> Any:
    return shared_datadir / "complex.prm7"


@pytest.fixture
def complex_rst7_path(shared_datadir: Any) -> Any:
    return shared_datadir / "complex.rst7"


class TestSuiteMinimise:
    # Parameterise, but skip OpenMM as this is failing
    @pytest.mark.parametrize("engine", BSSEngine)
    def test_biosimspace_minimise(
        self,
        temp_working_dir: Any,
        complex_prm7_path: Any,
        complex_rst7_path: Any,
        engine: BSSEngine,
    ) -> None:
        """Test the BioSimSpace minimisation node."""

        rig = TestRig(globals()[f"Minimise{engine.class_name}"])
        res = rig.setup_run(
            inputs={"inp": [[complex_prm7_path, complex_rst7_path]]},
            parameters={"steps": 10},
        )
        output = res["out"].get()
        # Get the file name from the path
        file_names = {f.name for f in output}
        assert file_names == {"bss_system.gro", "bss_system.top"}
