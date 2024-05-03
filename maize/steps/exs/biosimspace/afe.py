"""
Functionality for running alchemical bining free energy 
calculations using BioSimSpace.
"""

# pylint: disable=import-outside-toplevel, import-error

import csv
import pickle as pkl
import threading
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
import pytest

from maize.core.interface import (
    FileParameter,
    Input,
    MultiInput,
    MultiOutput,
    Output,
    Parameter,
    Suffix,
)
from maize.core.node import JobResourceConfig, Node
from maize.utilities.testing import TestRig

from ._base import _BioSimSpaceBase
from ._utils import create_engine_specific_nodes, get_ligand_smiles, mark_ligand_for_decoupling
from .enums import BSSEngine, Ensemble
from .exceptions import BioSimSpaceNullSystemError
from .production import _ProductionBase

_ENGINES = [BSSEngine.SOMD, BSSEngine.GROMACS]
"""The supported engines for alchemical binding free energy calculations."""

__all__ = [f"AFE{engine.class_name}" for engine in _ENGINES]
__all__.extend(
    ["SaveAFEResult", "AFEResult", "CollectAFEResults", "GenerateBoreschRestraintGromacs"]
)


# class DecoupleMolecules(_BioSimSpaceBase):
#     """
#     Mark a ligand to be decoupled for a BioSimSpace ABFE calculation and
#     return the files to create a perturbable system. The ligand must
#     be called "LIG".

#     Notes
#     -----
#     Install with `mamba create -f env.yaml`.

#     References
#     ----------
#     L. O. Hedges et al., JOSS, 2019, 4, 1831.
#     L. O. Hedges et al., LiveCoMS, 2023, 5, 2375–2375.
#     """

#     # We must define the engine for the base class
#     bss_engine = BSSEngine.NONE

#     # Parameters
#     decouple: Parameter[bool] = Parameter(default=True)
#     """
#     Whether to couple the intra-molecule forces. Setting to ``False``
#     means the intra-molecular forces will *not* be changed by the lambda
#     thereby decouple the molecule instead of annihilate it.
#     """

#     def run(self) -> None:
#         import BioSimSpace.Sandpit.Exscientia as BSS

#         # Load the input
#         system = self._load_input(sandpit=True)

#         # Find the ligand - assume that it's called "LIG"
#         try:
#             lig = system.search("resname LIG").molecules()[0]
#         except IndexError:
#             raise ValueError("No ligand called 'LIG' found in the input system.")

#         # Decouple the ligand
#         lig_decoupled = BSS.Align.decouple(lig)
#         system.updateMolecule(system.getIndex(lig), lig_decoupled)

#         # Save the output
#         self._save_pertrubable_output(system)


class _GenerateBoreschRestraintBase(_ProductionBase, ABC):
    """
    Run a short production simulation and analyse this
    to select Boresch restraints for an absolute
    binding free energy calculation.

    Notes
    -----
    Install with `mamba create -f env.yaml`.

    References
    ----------
    L. O. Hedges et al., JOSS, 2019, 4, 1831.
    L. O. Hedges et al., LiveCoMS, 2023, 5, 2375–2375.
    """

    # Input
    # TODO: This shouldn't be necessary as defined in the base class.
    # However, type info gets messed up if we remove this.
    # Figure out how to remove this
    # inp: Input[list[Path]] = Input(optional=True)
    # """
    # Paths to system input files. A topology and a coordinate
    # file are required. These can be in any of the formats
    # given by BSS.IO.fileFormats() e.g.:

    # gro87, grotop, prm7, rst rst7
    # """

    timestep: Parameter[float] = Parameter(default=2.0)
    """The integration timestep, in fs. Default = 2.0 fs."""

    runtime: Parameter[float] = Parameter(default=1.0)
    """The running time, in ns. Default = 1.0 ns."""

    # Parameters
    append_to_ligand_selection: Parameter[str] = Parameter(default="")
    """
    Appends the supplied string to the default atom selection which chooses
    the atoms in the ligand to consider as potential anchor points. The default
    atom selection is f'resname {ligand_resname} and not name H*'. Uses the
    mdanalysis atom selection language. For example, 'not name O*' will result
    in an atom selection of f'resname {ligand_resname} and not name H* and not
    name O*'. Default = "".
    """

    receptor_selection_str: Parameter[str] = Parameter(default="protein and name CA C N")
    """
    The selection string for the atoms in the receptor to consider
    as potential anchor points. The default atom selection is
    'protein and name CA C N'. Uses the mdanalysis atom selection
    language. Default = 'protein and name CA C N'.
    """

    force_constant: Parameter[float] = Parameter(default=0.0)
    """
    The force constant to use for all restraints, in kcal mol-1 A-2 [rad-2].
    area will be converted to A-2 and exchanged for rad-2. If set to 0.0,
    the force constants are fit to fluctuations observed during the simulation.
    Default = 0.0.
    """

    cutoff: Parameter[float] = Parameter(default=10.0)
    """
    The greatest distance between ligand and receptor anchor atoms (A).
    Only affects behaviour when method == "BSS" Receptor anchors
    further than cutoff Angstroms from the closest ligand anchors will not
    be included in the search for potential anchor points.
    Default = 10.0 A.
    """

    restraint_idx: Parameter[int] = Parameter(default=0)
    """
    The index of the restraint from a list of candidate restraints ordered by
    suitability. Default = 0.
    """

    # Outputs
    boresch_restraint: MultiOutput[Path] = MultiOutput()
    """The selected Boresch restraint object."""

    def _run_process(self) -> None:
        """
        Run the BioSimSpace process returned by `_get_process`.
        using the engine specified by `bss_engine`. This should
        only be called within `run()`.
        """
        import BioSimSpace.Sandpit.Exscientia as BSS

        # Get the input
        system = self._load_input(sandpit=True)

        # Mark the ligand to be decoupled
        mark_ligand_for_decoupling(system, ligand_name="LIG")

        # Get the restraint search object
        restraint_search = BSS.FreeEnergy.RestraintSearch(
            system,
            protocol=self._get_protocol(),
            engine=self.bss_engine.name.lower(),
            work_dir=str(self.work_dir),
        )

        # Get the process
        process = restraint_search._process

        # Run the process and wait for it to finish
        self.logger.info(f"Running restraint search with {self.bss_engine.name}...")
        cmd = " ".join([process.exe(), process.getArgString()])
        # If GROMACS, set ntmp=1 to avoid domain decomposition which can hammer performance
        if self.bss_engine == BSSEngine.GROMACS:
            cmd += " -ntmpi 1"
        options = JobResourceConfig(custom_attributes={"gres": "gpu:1", "mem": "24GB"})
        self.run_command(cmd, batch_options=options, prefer_batch=True)
        output_system = process.getSystem(block=True)
        # BioSimSpace sometimes returns None, so we need to check
        if output_system is None:
            raise BioSimSpaceNullSystemError("The output system is None.")

        # Save the output
        self._save_output(output_system, sandpit=True)

        # Analyse the trajectory to get the restraint
        restraint = restraint_search.analyse(method="BSS", block=True)

        # Save a restraint string to the directory for debugging purposes
        with open(self.work_dir / "restraint.txt", "w") as f:
            f.write(restraint.toString(engine="SOMD"))

        # Pickle the restraint object and send it to the output
        restraint_path = self.work_dir / "restraint.pkl"
        with open(restraint_path, "wb") as f:
            pkl.dump(restraint, f)

        for boresch_out in self.boresch_restraint:
            boresch_out.send(restraint_path)

        # Dump the data
        self._dump_data()

    def _get_protocol(self) -> "BSS.Protocol._protocol.Protocol":
        import BioSimSpace.Sandpit.Exscientia as BSS

        pressure = (
            None
            if Ensemble(self.ensemble.value) == Ensemble.NVT
            else self.pressure.value * BSS.Units.Pressure.atm
        )

        return BSS.Protocol.Production(
            timestep=self.timestep.value * BSS.Units.Time.femtosecond,
            runtime=self.runtime.value * BSS.Units.Time.nanosecond,
            temperature=self.temperature.value * BSS.Units.Temperature.kelvin,
            pressure=pressure,
            tau_t=self.thermostat_time_constant.value * BSS.Units.Time.picosecond,
            report_interval=self.report_interval.value,
            restart_interval=self.restart_interval.value,
            restraint=self.restraint.value if self.restraint.value else None,
            force_constant=self.force_constant.value,
        )


create_engine_specific_nodes(_GenerateBoreschRestraintBase, __name__, _ENGINES)


@dataclass
class AFEResult:
    """A class for storing the results of an alchemical binding free energy calculation."""

    smiles: str
    """The SMILES string of the ligand."""
    dg: np.ndarray
    """The free energy changes for the alchemical transformation for each repeat."""
    error: np.ndarray
    """The error in the free energy change for each repeat."""
    pmf: list[pd.Series] | None
    """List of potentials of mean force (PMFs) for the alchemical transformation."""
    overlap: list[pd.Series] | None
    """The overlap matrices between windows for the alchemical transformation."""

    # Allow addition or subtraction of two results if they have the same SMILES
    # Note that this removes the pmf and overlap information
    def __add__(self, other: "AFEResult") -> "AFEResult":
        if self.smiles != other.smiles:
            raise ValueError("Cannot add two results with different SMILES.")
        return AFEResult(
            smiles=self.smiles,
            dg=self.dg + other.dg,
            error=np.sqrt(self.error**2 + other.error**2),
            pmf=None,
            overlap=None,
        )

    # Define subtraction as addition with the negative of the other result
    def __sub__(self, other: "AFEResult") -> "AFEResult":
        return self + AFEResult(
            smiles=other.smiles,
            dg=-other.dg,
            error=other.error,
            pmf=None,
            overlap=None,
        )

    # Allow use of sum() function
    def __radd__(self, other: "AFEResult") -> "AFEResult":
        if other == 0:
            return self
        return self.__add__(other)

    def append_repeat(self, other: "AFEResult") -> None:
        if self.smiles != other.smiles:
            raise ValueError("Cannot append two results with different SMILES.")
        self.dg = np.append(self.dg, other.dg)
        self.error = np.append(self.error, other.error)
        self.pmf.append(other.pmf)
        self.overlap.append(other.overlap)

    def __len__(self) -> int:
        return len(self.dg)


class SaveAFEResult(Node):
    """Save an AFE result object to a CSV. The PMF and overlap information is discarded."""

    # Often, we will be writing to the same file from multiple nodes, so we need to lock
    lock = threading.Lock()

    inp: Input[AFEResult] = Input()
    """An alchemical free energy result, or a list of results."""

    file: FileParameter[Annotated[Path, Suffix("csv")]] = FileParameter(
        exist_required=False, default=Path("afe_results.csv")
    )
    """Output CSV location"""

    def run(self) -> None:
        result = self.inp.receive()
        # Aquire the lock before writing to the file
        with self.lock:
            with open(self.file.filepath, "a") as out:
                writer = csv.writer(out, delimiter=",", quoting=csv.QUOTE_MINIMAL)
                # Only write header if it's empty
                if not self.file.filepath.exists() or self.file.filepath.stat().st_size == 0:
                    writer.writerow(["smiles", "repeat_no", "dg", "error"])
                for i in range(len(result)):
                    # Get the repeat number by checking if there are already lines with the current smiles
                    # present
                    writer.writerow([result.smiles, i + 1, result.dg[i], result.error[i]])


class CollectAFEResults(Node):
    """Collect multiple ABFE results and sum them."""

    inp: MultiInput[AFEResult] = MultiInput()
    """The ABFE results to sum."""

    out: Output[AFEResult] = Output()
    """The summed ABFE result."""

    def run(self) -> None:
        afe_res = [self.inp[i].receive() for i in range(len(self.inp))]
        self.out.send(sum(afe_res))


class _AFEBase(_BioSimSpaceBase, ABC):
    """
    Abstract base class for BioSimSpace alchemical free energy
    nodes. All subclasses should set the biosimspace_engine attribute,
    and the required_callables attribute will be set automatically.

    Notes
    -----
    Install with `mamba create -f env.yaml`.

    References
    ----------
    L. O. Hedges et al., JOSS, 2019, 4, 1831.
    L. O. Hedges et al., LiveCoMS, 2023, 5, 2375–2375.
    """

    boresch_restraint: Parameter[Path] = Input(optional=True)
    """
    The Boresch restraint to use, if performing an absolute binding free energy
    calculation. This should be a pickle of the BioSimSpace Restraint
    object.
    """

    # Parameters
    n_replicates: Parameter[int] = Parameter(default=5)
    """The number of replicate calculations to run for each ligand. Default = 5."""

    lam_vals: Parameter[list[float] | pd.Series] = Parameter(default=[0.1 * i for i in range(11)])
    """
    The lambda values to use for the alchemical free energy calculation. Default = [0.0, 0.1, ..., 1.0].
    """

    timestep: Parameter[float] = Parameter(default=2.0)
    """The integration timestep, in fs. Default = 2.0 fs."""

    runtime: Parameter[float] = Parameter(default=4.0)
    """The running time, in ns. Default = 4.0 ns."""

    temperature: Parameter[float] = Parameter(default=300.0)
    """The temperature, in K. Default = 300.0 K."""

    pressure: Parameter[float] = Parameter(default=1)
    """The pressure, in atm. Default = 1 atm."""

    tau_t: Parameter[float] = Parameter(default=1.0)
    """The thermostat time constant, in ps. Default = 1.0 ps."""

    report_interval: Parameter[int] = Parameter(default=200)
    """The frequency at which statistics are recorded. (In integration steps.) Default = 200."""

    restart_interval: Parameter[int] = Parameter(default=1000)
    """
    The frequency at which restart configurations and trajectory frames are saved. 
    (In integration steps.) Default = 1000."""

    perturbation_type: Parameter[str] = Parameter(default="full")
    """
    The type of perturbation to perform. Options are:
    "full" : A full perturbation of all terms (default option).
    "discharge_soft" : Perturb all discharging soft atom charge terms (i.e. value->0.0).
    "vanish_soft" : Perturb all vanishing soft atom LJ terms (i.e. value->0.0).
    "flip" : Perturb all hard atom terms as well as bonds/angles.
    "grow_soft" : Perturb all growing soft atom LJ terms (i.e. 0.0->value).
    "charge_soft" : Perturb all charging soft atom LJ terms (i.e. 0.0->value).
    "restraint" : Perturb the receptor-ligand restraint strength by linearly
                scaling the force constants (0.0->value).
    "release_restraint" : Used with multiple distance restraints to release all
                          restraints other than the "permanent" one when the ligand
                          is fully decoupled. Note that lambda = 0.0 is the fully
                          released state, and lambda = 1.0 is the fully restrained
                          state (i.e. 0.0 -> value).

    Currently perturubation_type != "full" is only supported by
    BioSimSpace.Process.Somd.

    Default = "full".
    """

    dg_sign: Parameter[Literal[-1, 1]] = Parameter(default=1)
    """
    If set to -1, the sign of the free energy change will be flipped.
    This is useful when integrating ABFE stages into an entire ABFE calculation.
    Default = 1.
    """

    apply_restraint_correction: Parameter[bool] = Parameter(default=False)
    """
    If set to True, the restraint correction will be applied to the free energy
    change. This is useful when integrating ABFE stages into an entire ABFE calculation.
    Note that the correction is for releasing the restraint to the standard state volume,
    and its sign is reversed if dg_sign is set to -1. Default = False.
    """

    runtime: Parameter[float] = Parameter(default=1.0)
    """The running time, in ns. Default = 1.0 ns."""

    temperature: Parameter[float] = Parameter(default=300.0)
    """The temperature, in K. Default = 300.0 K."""

    ensemble: Parameter[Literal["NVT", "NPT"]] = Parameter(default="NVT")
    """The ensemble to use. Default = NPT."""

    pressure: Parameter[float] = Parameter(default=1)
    """The pressure, in atm. Default = 1 atm."""

    thermostat_time_constant: Parameter[float] = Parameter(default=1.0)
    """Time constant for thermostat coupling, in ps. Default = 1.0 ps."""

    report_interval: Parameter[int] = Parameter(default=100)
    """The frequency at which statistics are recorded. (In integration steps.). Default = 100."""

    restart_interval: Parameter[int] = Parameter(default=500)
    """
    The frequency at which restart configurations and trajectory
    frames are saved. (In integration steps.) Default = 500.
    """

    restart: Parameter[bool] = Parameter(default=False)
    """Whether this is a continuation of a previous simulation. Default = False."""

    estimator: Parameter[str] = Parameter(default="MBAR")
    """
    The estimator to use when calculating the free energy. Can be one of
    "MBAR" or "TI". Default = "MBAR".
    """

    # Output
    out: Output[AFEResult] = Output()
    """The results of the alchemical free energy calculation."""

    def run(self) -> None:
        self._run_process()

    # Overwrite the base class _run_process method as AFE calculations are
    # fairly different.
    def _run_process(self) -> None:
        """
        Run the BioSimSpace process returned by `_get_process`.
        using the engine specified by `bss_engine`. This should
        only be called within `run()`.
        """
        import BioSimSpace.Sandpit.Exscientia as BSS

        # Get the perturbable system
        sys = self._load_input(sandpit=True)
        pert_sys = mark_ligand_for_decoupling(sys, ligand_name="LIG")

        # Get the restraint, if it exists
        if self.boresch_restraint.is_set:
            with open(self.boresch_restraint.receive(), "rb") as f:
                restraint = pkl.load(f)
            # Writing and saving a BSS produces formally different systems, but
            # the restraint system and the system used for the calculation must be the same.
            # Therefore, use the system from the restraint file.
            pert_sys = restraint._system
        else:
            restraint = None

        # Loop for the number of replicates
        overall_afe_result = AFEResult(
            smiles=get_ligand_smiles(sys, ligand_name="LIG"),
            dg=np.array([]),
            error=np.array([]),
            pmf=[],
            overlap=[],
        )
        for i in range(self.n_replicates.value):
            self.logger.info(f"Running replicate {i + 1} of {self.n_replicates.value}...")

            # Create the alchemical free energy object.
            protocol = self._get_protocol()
            calculation = BSS.FreeEnergy.AlchemicalFreeEnergy(
                system=pert_sys,
                protocol=protocol,
                work_dir=str(self.work_dir),
                engine=self.bss_engine.name.lower(),
                gpu_support=True,
                restraint=restraint,
                estimator=self.estimator.value,
                extra_options=self._get_extra_options(self.bss_engine),
            )

            # Set up the calculation processes
            calculation._initialise_runner(pert_sys)

            # Get all of the process commands and working directories
            self.logger.info(
                f"Running {protocol.__class__.__name__} with {self.bss_engine.name}..."
            )
            cmds = [" ".join([p.exe(), p.getArgString()]) for p in calculation._runner.processes()]
            # For SOMD, we need to ensure that the partition is CUDA and not CPU. CUDA_VISIBLE_DEVICES
            # is only set in the run_multi method, hence BioSimSpace will set the partition to CPU.
            if self.bss_engine == BSSEngine.SOMD:
                cmds = [" ".join(cmd.split(" ")[:-2]) + " -p CUDA" for cmd in cmds]
            # If GROMACS, set ntmp=1 to avoid domain decomposition which can hammer performance
            if self.bss_engine == BSSEngine.GROMACS:
                cmds = [cmd + " -ntmpi 1" for cmd in cmds]
            work_dirs = [str(p._work_dir) for p in calculation._runner.processes()]

            # Run all of the lambda windows, ensuring that they
            # get one GPU each
            options = JobResourceConfig(custom_attributes={"gres": "gpu:1", "mem": "24GB"})
            self.run_multi(cmds, work_dirs, batch_options=options)

            # Analyse the stage
            self.logger.info("Analysing the stage...")
            self.logger.debug(f"dg_multiplier: {self.dg_sign.value}")
            dg_multiplier = self.dg_sign.value
            smiles = get_ligand_smiles(sys, ligand_name="LIG")
            pmf, overlap = calculation.analyse()
            dg = pmf[-1][1].value() * dg_multiplier
            dg_error = pmf[-1][2].value()
            self.logger.info(f"Free energy change for {smiles}: {dg} +/- {dg_error} kcal/mol")
            # Make sure to apply the correction if necessary
            if self.apply_restraint_correction.value and restraint is not None:
                self.logger.info(
                    f"Applying the restraint correction: {restraint.getCorrection().value() * dg_multiplier} kcal/mol"
                )
                dg += restraint.getCorrection().value() * dg_multiplier

            afe_result = AFEResult(
                smiles=smiles,
                dg=np.array([dg]),
                error=np.array([dg_error]),
                pmf=pmf,
                overlap=overlap,
            )
            overall_afe_result.append_repeat(afe_result)

        # Return the free energy change
        self.out.send(overall_afe_result)

        # Save the current AFE result to the work directory
        with open(self.work_dir / "afe_result.pkl", "wb") as f:
            pkl.dump(overall_afe_result, f)

        # Dump required data
        self._dump_data()

    def _get_protocol(self) -> "BSS.Protocol._protocol.Protocol":
        import BioSimSpace.Sandpit.Exscientia as BSS

        # The lam parameter has to be different for SOMD and GROMACS
        lam_map = {
            BSSEngine.SOMD: 0.0,
            BSSEngine.GROMACS: pd.Series(data={"fep": 0.0}),
        }

        return BSS.Protocol.FreeEnergy(
            lam=lam_map[self.bss_engine],
            lam_vals=self.lam_vals.value,
            min_lam=0.0,
            max_lam=1.0,
            timestep=self.timestep.value * BSS.Units.Time.femtosecond,
            runtime=self.runtime.value * BSS.Units.Time.nanosecond,
            temperature=self.temperature.value * BSS.Units.Temperature.kelvin,
            pressure=self.pressure.value * BSS.Units.Pressure.atm,
            tau_t=self.tau_t.value * BSS.Units.Time.picosecond,
            report_interval=self.report_interval.value,
            restart_interval=self.restart_interval.value,
            perturbation_type=self.perturbation_type.value,
        )

    def _get_extra_options(self, engine: BSSEngine) -> dict | None:
        """Get extra options to pass to the BSS AlchemicalFreeEnergy object."""
        if engine == BSSEngine.SOMD:
            # Work out a reasonable number of cycles as the default is terrible (
            # produces far too many cycles). Once every 25000 steps is a good
            # starting point.
            total_steps = round(
                self.runtime.value * 1e6 / self.timestep.value
            )  # runtime in ns, timestep in fs
            if total_steps < 25000:  # Very short run - just run one cycle
                ncycles = 1
                nmoves = total_steps
            else:
                ncycles = total_steps // 25000
                nmoves = 25000
            if ncycles * nmoves < total_steps:
                self.logger.warning(
                    f"The number of steps is not divisible by 25000. The rumtime will be {ncycles * nmoves * self.timestep.value / 1e6} ns"
                    f" rather than {self.runtime.value} ns."
                )

            extra_options = {
                "ncycles": ncycles,
                "nmoves": nmoves,
                "hydrogen mass repartitioning factor": 3.0,
                "cutoff distance": "12 * angstrom",  # As we use RF
                "integrator": "langevinmiddle",
                "inverse friction": "1 * picosecond",
                "thermostat": False,  # Handled by langevin integrator
            }

        elif engine == BSSEngine.GROMACS:
            extra_options = None

        return extra_options


create_engine_specific_nodes(_AFEBase, __name__, _ENGINES)


@pytest.fixture
def restraint_path(shared_datadir: Any) -> Any:
    return shared_datadir / "restraint.pkl"


class TestSuiteAFE:
    # Note that currently SOMD does not work for restraint search
    @pytest.mark.parametrize("engine", [BSSEngine.GROMACS])
    def test_restraint_search(
        self,
        temp_working_dir: Any,
        complex_prm7_path: Any,
        complex_rst7_path: Any,
        engine: BSSEngine,
    ) -> None:
        """Test the BioSimSpace GenerateBoresch node."""

        rig = TestRig(globals()[f"GenerateBoreschRestraint{engine.class_name}"])
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

        # Check that we have a restraint file
        restr_pkl = res["boresch_restraint"].get()
        assert restr_pkl.name == "restraint.pkl"

        # Read the restraint txt file and check that it's reasonable
        # Check that the dumping worked
        # Get the most recent directory in the dump folder
        dump_output_dir = sorted(dump_dir.iterdir())[-1]
        with open(dump_output_dir / "restraint.txt", "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            for component in ["anchor_points", "equilibrium_values", "force_constants"]:
                assert component in lines[0]

    @pytest.mark.parametrize("engine", _ENGINES)
    def test_abfe_calcs(
        self,
        temp_working_dir: Any,
        complex_prm7_path: Any,
        complex_rst7_path: Any,
        restraint_path: Any,
        engine: BSSEngine,
    ) -> None:
        """Test the afe nodes by setting up absolute binding free energy calculations."""

        rig = TestRig(globals()[f"AFE{engine.class_name}"])
        dump_dir = Path().absolute().parents[1] / "dump"
        res = rig.setup_run(
            inputs={
                "inp": [[complex_prm7_path, complex_rst7_path]],
                "boresch_restraint": [restraint_path],
            },
            parameters={
                "runtime": 0.001,
                "dump_to": dump_dir,
                "estimator": "TI",
                "n_replicates": 2,
            },
        )

        # Check that we have all the expected outputs. Note that values will
        # differ from run to run and engine to engine
        afe_result = res["out"].get()
        assert (
            afe_result.smiles
            == "COc1ccc(C2=NC(c3ccc(Cl)cc3)C(c3ccc(Cl)cc3)N2C(=O)N2CCNC(=O)C2)c(OC(C)C)c1"
        )
        # Check relative magnitdues of errors and results are similar
        assert all(afe_result.dg / 2 > afe_result.error)
        assert len(afe_result.pmf[0]) == 11

        # Check that the dumping worked and that we have several lambda
        # directories
        dump_output_dir = sorted(dump_dir.iterdir())[-1]
        lam_dirs = [d for d in dump_output_dir.iterdir() if "lambda" in d.name]
        assert len(lam_dirs) == 11

    def test_afe_result(self, temp_working_dir: Any) -> None:
        """Check that we can add and subtract AFE results as expected."""

        result1 = AFEResult(
            "c1ccccc1", np.array([1.0]), np.array([0.1]), [[0.0, 0.5, 1.0]], [[0.0, 0.5, 1.0]]
        )
        result2 = AFEResult(
            "C#N", np.array([1.0]), np.array([0.1]), [[0.0, 0.5, 1.0]], [[0.0, 0.5, 1.0]]
        )

        # Check that we can add and subtract results
        sum11 = result1 + result1
        assert sum11.smiles == "c1ccccc1"
        assert pytest.approx(sum11.dg) == 2.0
        assert pytest.approx(sum11.error) == 0.1414213562373095

        diff11 = result1 - result1
        assert diff11.smiles == "c1ccccc1"
        assert pytest.approx(diff11.dg) == 0.0
        assert pytest.approx(diff11.error) == 0.1414213562373095

        with pytest.raises(ValueError):
            result1 + result2

        # Check that we an append results
        result1.append_repeat(result1)
        assert result1.dg.shape == (2,)
        assert result1.error.shape == (2,)
        assert len(result1.pmf) == 2
        assert len(result1.overlap) == 2

    def test_collect_afe_results(self, temp_working_dir: Any) -> None:
        """Test the CollectAFEResults node."""

        rig = TestRig(CollectAFEResults)
        res = rig.setup_run(
            inputs={
                "inp": [
                    AFEResult("c1ccccc1", 1.0, 0.1, [], []),
                    AFEResult("c1ccccc1", 1.0, 0.1, [], []),
                ]
            }
        )
        output = res["out"].get()
        assert output.smiles == "c1ccccc1"
        assert pytest.approx(output.dg) == 2.0
        assert pytest.approx(output.error) == 0.1414213562373095

    def test_save_afe_results(self, temp_working_dir: Any) -> None:
        """Test the SaveAFEResult node."""

        rig = TestRig(SaveAFEResult)
        csv_path = Path() / "results.csv"
        res = rig.setup_run(
            inputs={
                "inp": [
                    AFEResult(
                        "COc1ccc(C2=NC(c3ccc(Cl)cc3)C(c3ccc(Cl)cc3)N2C(=O)N2CCNC(=O)C2)c(OC(C)C)c1",
                        np.array([1.0]),
                        np.array([0.1]),
                        [],
                        [],
                    )
                ]
            },
            parameters={"file": csv_path},
        )

        # Check that the output file has been created
        assert csv_path.exists()
        lines = [l for l in csv_path.read_text().split("\n") if l != ""]
        assert len(lines) == 2
        assert lines[0] == "smiles,repeat_no,dg,error"
        assert (
            lines[1]
            == "COc1ccc(C2=NC(c3ccc(Cl)cc3)C(c3ccc(Cl)cc3)N2C(=O)N2CCNC(=O)C2)c(OC(C)C)c1,1,1.0,0.1"
        )

    # def test_decouple_molecules(
    #     self,
    #     temp_working_dir: Any,
    #     complex_prm7_path: Any,
    #     complex_rst7_path: Any,
    # ) -> None:
    #     """Test the BioSimSpace decouple molecules node."""

    #     rig = TestRig(DecoupleMolecules)
    #     res = rig.setup_run(
    #         inputs={"inp": [[complex_prm7_path, complex_rst7_path]]},
    #         parameters={},
    #     )
    #     output = res["out"].get()
    #     # Get the file name from the path
    #     file_names = {f.name for f in output}
    #     assert file_names == {
    #         "bss_system0.prm7",
    #         "bss_system0.rst7",
    #         "bss_system1.prm7",
    #         "bss_system1.rst7",
    #     }
