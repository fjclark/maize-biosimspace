"""Functionality for equilibrating a system using BioSimSpace"""

# pylint: disable=import-outside-toplevel, import-error

from abc import ABC
from pathlib import Path
from typing import Any

import pytest

from maize.core.interface import Parameter
from maize.utilities.testing import TestRig

from ._base import _BioSimSpaceBase
from ._utils import create_engine_specific_nodes
from .enums import BSSEngine, Ensemble

_ENGINES = [
    Engine
    for Engine in BSSEngine
    if Engine not in [BSSEngine.OPENMM, BSSEngine.TLEAP, BSSEngine.NAMD, BSSEngine.NONE]
]
"""The supported engines for equilibration."""

__all__ = [f"Equilibrate{engine.class_name}" for engine in _ENGINES]


class _EquilibrateBase(_BioSimSpaceBase, ABC):
    """
    Abstract base class for BioSimSpace equilibration nodes.
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

    # Parameters
    timestep: Parameter[float] = Parameter(default=2.0)
    """The integration timestep, in fs."""

    runtime: Parameter[float] = Parameter(default=0.2)
    """The running time, in ns."""

    temperature_start: Parameter[float] = Parameter(default=300.0)
    """The starting temperature, in K."""

    temperature_end: Parameter[float] = Parameter(default=300.0)
    """The ending temperature, in K."""

    ensemble: Parameter[Ensemble] = Parameter(default=Ensemble.NVT)
    """The ensemble to use for the equilibration."""

    pressure: Parameter[float] = Parameter(default=1)
    """
    The pressure, in atm. This is ignored if the ensemble is NVT."""

    thermostat_time_constant: Parameter[float] = Parameter(default=1.0)
    """Time constant for thermostat coupling, in ps."""

    report_interval: Parameter[int] = Parameter(default=100)
    """The frequency at which statistics are recorded. (In integration steps.)"""

    restart_interval: Parameter[int] = Parameter(default=500)
    """
    The frequency at which restart configurations and trajectory
    frames are saved. (In integration steps.)
    """

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

    save_name: Parameter[Path] = Parameter(optional=True)
    """
    If supplied, the output files will be saved with
    this name. E.g., if set to Path("output"), the output
    files will be output.rst7 and output.prm7.
    """

    def run(self) -> None:
        self._run_process()

    def _get_protocol(self) -> "BSS.Protocol._protocol.Protocol":
        import BioSimSpace as BSS

        pressure = (
            None
            if self.ensemble.value == Ensemble.NVT
            else self.pressure.value * BSS.Units.Pressure.atm
        )

        return BSS.Protocol.Equilibration(
            timestep=self.timestep.value * BSS.Units.Time.femtosecond,
            runtime=self.runtime.value * BSS.Units.Time.nanosecond,
            temperature_start=self.temperature_start.value * BSS.Units.Temperature.kelvin,
            temperature_end=self.temperature_end.value * BSS.Units.Temperature.kelvin,
            pressure=pressure,
            thermostat_time_constant=self.thermostat_time_constant.value
            * BSS.Units.Time.picosecond,
            report_interval=self.report_interval.value,
            restart_interval=self.restart_interval.value,
            restraint=self.restraint.value if self.restraint.value else None,
            force_constant=self.force_constant.value,
        )


create_engine_specific_nodes(_EquilibrateBase, __name__, _ENGINES, create_exposed_workflows=True)


class TestSuiteEquilibrate:
    @pytest.mark.parametrize("engine", _ENGINES)
    def test_biosimspace_equilibrate(
        self,
        temp_working_dir: Any,
        complex_prm7_path: Any,
        complex_rst7_path: Any,
        engine: BSSEngine,
    ) -> None:
        """Test the BioSimSpace minimisation node."""

        rig = TestRig(globals()[f"Equilibrate{engine.class_name}"])
        dump_dir = Path().absolute().parents[1] / "dump"
        res = rig.setup_run(
            inputs={"inp": [[complex_prm7_path, complex_rst7_path]]},
            parameters={"runtime": 0.001, "dump_to": dump_dir},
        )
        output = res["out"].get()
        # Get the file name from the path
        file_names = {f.name for f in output}
        assert all(f.startswith("bss_system") for f in file_names)
        assert all(f.endswith(".prm7") or f.endswith(".rst7") for f in file_names)

        # Check that the dumping worked
        # Get the most recent directory in the dump folder
        dump_output_dir = sorted(dump_dir.iterdir())[-1]
        assert any(f.name.startswith("bss_system") for f in dump_output_dir.iterdir())
        assert any(f.name.endswith(".prm7") for f in dump_output_dir.iterdir())
        assert any(f.name.endswith(".rst7") for f in dump_output_dir.iterdir())
