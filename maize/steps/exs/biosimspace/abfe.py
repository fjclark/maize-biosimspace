"""A single large node for running ABFE calculations on many compounds in parallel."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence, cast

import numpy as np
import pytest
import scipy.stats as stats

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
from maize.utilities.chem import Isomer, load_sdf_library
from maize.utilities.testing import TestRig


@dataclass
class ABFEResult:
    """A class for storing the results of an ABFE calculation."""

    smiles: str
    """The SMILES of the molecule."""

    inchi_key: str
    """The InChI key of the molecule."""

    dg_array: np.ndarray
    """The free energy difference in kcal/mol for each repeat."""

    error_array: np.ndarray
    """The error in the free energy difference in kcal/mol for each repeat."""

    @property
    def dg(self) -> float:
        """The mean free energy difference in kcal/mol."""
        return np.mean(self.dg_array)

    @property
    def ci_95(self) -> float:
        """
        The 95% t-based confidence interval for the free energy difference, calcualted
        from the inter-replica standard deviation.
        """
        return (
            stats.t.interval(
                0.95, len(self.dg_array) - 1, loc=self.dg, scale=stats.sem(self.dg_array)
            )[1]
            - self.dg
        )

    def __len__(self) -> int:
        """The number of repeats."""
        return len(self.dg_array)

    def __str__(self) -> str:
        return f"{self.inchi}: {self.dg:.2f} ± {self.ci_95:.2f} kcal/mol"


class ABFEResultEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ABFEResult):
            return {
                "inchi": obj.inchi_key,
                "smiles": obj.smiles,
                "dg_array": obj.dg_array.tolist(),
                "error_array": obj.error_array.tolist(),
                "overall_dg": obj.dg,
                "ci_95": obj.ci_95,
                "unit": "kcal/mol",
            }
        return super().default(obj)


class BssAbfe(Node):
    """A single large node for running ABFE calculations on many isomers in parallel."""

    required_callables = ["gmx", "somd-freenrg"]
    """
    somd-freenrg
        Required for running the ABFE calculations using sire and OpenMM. Will
        be available in the environment through sire.

    gmx
        Used for the equilibration stages. Will need to be installed separately.
    """

    required_packages = ["BioSimSpace"]
    """
    BioSimSpace
        Required to set up all calculations.
    """

    inp: Input[list[Isomer]] = Input()
    """Molecules to run ABFE calculations on."""

    inp_protein: Input[Annotated[Path, Suffix("pdb")]] = Input(cached=True)
    """Protein structure to use for the ABFE calculations."""

    inp_cofactor_sdf: Input[Annotated[Path, Suffix("pdb")]] = Input(optional=True)
    """"
    Path to the sdf file of the cofactor to use in the ABFE calculations. This
    will be parameterised in the same way as the ligands.
    """

    out: Output[dict[str, ABFEResult]] = Output()
    """A dictionary of InChI keys to ABFE results."""

    dump_to: FileParameter[Path] = FileParameter(optional=True)
    """A directory to dump all generated data to."""

    temperature: Parameter[float] = Parameter(default=298.15)
    """The temperature in K to run all simulations at. Default is 298.15 K."""

    protein_force_field: Parameter[Literal["ff99", "ff99SB", "ff99SBildn", "ff14SB"]] = Parameter(
        default="ff14SB"
    )
    """
    The force field to use for the protein. One of "ff99", "ff99SB", "ff99SBildn", "ff14SB".
    Default is "ff14SB".
    """

    ligand_force_field: Parameter[
        Literal["gaff", "gaff2", "openff_unconstrained-1.0.0", "openff_unconstrained-2.0.0"]
    ] = Parameter(default="gaff2")
    """
    The force field to use for the ligands. One of "gaff", "gaff2", 
    "openff_unconstrained-1.0.0", "openff_unconstrained-2.0.0". Default is "gaff2".
    """

    water_model: Parameter[Literal["tip3p", "tip4p", "tip5p", "spc", "spce"]] = Parameter(
        default="tip3p"
    )
    """
    The water model to use in the simulations. One of "tip3p", "tip4p", "tip5p", "spc", "spce".
    Default is "tip3p".
    """

    ion_conc: Parameter[float] = Parameter(default=0.15)
    """The ion concentration in M of NaCl to use in the simulations. Default is 0.15 M."""

    equilibration_protocol: Parameter[Literal["fast", "slow"]] = Parameter(default="fast")
    """Whether to use a fast or slow equilibration protocol. One of "fast" or "slow". Default is "fast"."""

    run_time_ensemble_equilibration: Parameter[float] = Parameter(default=0.1)
    """
    The time in ns to run each ensemble equilibration for. These are intended to be short initial
    production MD runs, run for each repeat, to generate diverse starting structures. Default is 0.1 ns.
    """

    run_time_abfe: Parameter[float] = Parameter(default=0.1)
    """The time in ns to run each window of the ABFE calculations for. Default is 0.1 ns."""

    n_repeats: Parameter[int] = Parameter(default=3)
    """The number of repeats to run for each ligand. Default is 3."""

    lambda_schedule_free: Parameter[dict[str, Sequence[float]]] = Parameter(
        default={
            "discharge": [0.0, 0.222, 0.447, 0.713, 1.0],
            "vanish": [
                0.0,
                0.026,
                0.055,
                0.09,
                0.126,
                0.164,
                0.202,
                0.239,
                0.276,
                0.314,
                0.354,
                0.396,
                0.437,
                0.478,
                0.518,
                0.559,
                0.606,
                0.668,
                0.762,
                1.0,
            ],
        },
    )
    """
    The lambda schedule to use for the free leg of the  ABFE calculations. This should be a dictionary
    containing lists of lambda values for each stage type. The default was optimised for the current
    soft-core options using a ~ 30 atom ligand, but overlap is still sufficient for ~ 70 atom ligands.
    """

    lambda_schedule_bound: Parameter[dict[str, Sequence[float]]] = Parameter(
        default={
            "restrain": [0.0, 1.0],
            "discharge": [0.0, 0.291, 0.54, 0.776, 1.0],
            "vanish": [
                0.0,
                0.026,
                0.054,
                0.083,
                0.111,
                0.14,
                0.173,
                0.208,
                0.247,
                0.286,
                0.329,
                0.373,
                0.417,
                0.467,
                0.514,
                0.564,
                0.623,
                0.696,
                0.833,
                1.0,
            ],
        },
    )
    """
    The lambda schedule to use for the bound leg of the  ABFE calculations. The default was optimised for the current
    soft-core options using a ~ 30 atom ligand, but overlap is still sufficient for ~ 70 atom ligands.
    """

    n_jobs: Parameter[int] = Parameter(default=300)
    """
    The number of jobs to run in parallel. Should be equal to the number of GPUs if local,
    or the number of batch job submissions. Default is 300.
    """

    production_timestep: Parameter[float] = Parameter(default=4.0)
    """
    The timestep in fs to use for the production simulations. 4 fs can be used due
    to hydrogen mass repartitioning. Default is 4 fs.
    """

    def run(self) -> None:

        # Create dictionaries to hold paths to the BSS systems
        # This holds BSS systems during equilibration
        self.bss_systems: dict[str, dict[str, Path | list[Path]]] = {"free": {}, "bound": {}}
        # This holds a list of length n_repeats of paths to the pre-equilibrated systems
        self._preequilibrated_systems: dict[str, dict[str, list[list[Path]]]] = {
            "free": {},
            "bound": {},
        }
        # This holds the paths to the ABFE directories
        self._abfe_dirs: dict[str, dict[str, dict[str, list[Path]]]] = {"free": {}, "bound": {}}
        # This holds information on failed isomers
        self.failed_isomers: dict[str, dict[str, Any]] = {}
        # Hold the results. This takes the form of a dictionary with isomer names as keys
        # and dictionaries as values. These dictionaries hold lists of the ABFE results
        # and per-run uncertainties.
        self.results: dict[str, ABFEResult] = {}

        self.isomers: list[Isomer] = []

        try:
            self._recieve_isomers()
            self._create_sdfs()
            self._parameterise()
            self._set_up_cofactor()
            self._assemble_complexes()
            self._solvate()
            self._minimise()
            self._equil_nvt_restrained()
            self._equil_nvt_backbone_restrained()
            self._equil_nvt_unrestrained()
            self._equil_npt_restrained()
            self._equil_npt_unrestrained()
            self._ensemble_equilibration()
            self._set_up_abfe()
            self._run_abfe()
            self._analyse_abfe()
            self._save_results_file()

            self.out.send(self.results)

            self.logger.info("ABFE calculations complete.")

        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            raise e

        finally:
            # Make sure to always dump the output
            self._dump_data()

    def _recieve_isomers(self) -> None:
        """Receive the isomers from the input."""
        self.isomers = self.inp.receive()

    def _create_sdfs(self) -> None:
        """Convert input isomers to sdfs."""

        self.logger.info("Converting isomers to sdfs...")
        for isomer in self.isomers:
            name = isomer.inchi
            sdf_path = self.work_dir / Path("load_input") / Path(name) / Path("isomer.sdf")
            sdf_path = sdf_path.absolute()
            sdf_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Saving {name} to {sdf_path}")
            isomer.to_sdf(sdf_path)
            self.bss_systems["free"][name] = sdf_path

    def _parameterise(self) -> None:
        """Parameterise the ligands and protein. This leaves the free systems as parameterised
        ligands in vacuum, and the bound systems as the parameterised protein."""
        import BioSimSpace as BSS

        # Parameterise the ligands using run_command, as this is compute intensive
        # First, we need to set up all the directories
        self.logger.info("Parameterising ligands...")
        work_dirs = {}
        cmds = []
        for name, sdf_path in self.bss_systems["free"].items():
            work_dir = self.work_dir / Path("parameterise_ligands") / Path(name)
            work_dir.mkdir(parents=True, exist_ok=True)
            work_dirs[name] = work_dir

            # Create a python script to be run through run_command (so that we use the
            # desired scheduling system)
            param_script = (
                "import BioSimSpace as BSS\n"
                "from maize.steps.exs.biosimspace._utils import rename_lig\n"
                f"system = BSS.IO.readMolecules('{sdf_path}')\n"
                "system = system[0]\n"
                "param_mol = BSS.Parameters.parameterise(\n"
                "    system,\n"
                f"    forcefield='{self.ligand_force_field.value}',\n"
                f"    work_dir='{work_dir}',\n"
                ").getMolecule().toSystem()\n"
                "rename_lig(param_mol, new_name='LIG')\n"
                f'BSS.IO.saveMolecules("{work_dir}/slurm_out", param_mol, ["prm7", "rst7"])\n'
            )

            # Write the script to a file
            with open(work_dir / Path("param_script.py"), "w") as f:
                f.write(param_script)

            cmds.append(f"python {work_dir / Path('param_script.py')}")

        # Parameterise the ligands in parallel. We need lots of memory.
        options = JobResourceConfig(custom_attributes={"mem": "24GB"})
        self.run_multi(cmds, batch_options=options)

        # Check that output has been successfully generated and update the bss_systems dictionary
        for name, work_dir in work_dirs.items():
            output_paths = [work_dir / Path("slurm_out.prm7"), work_dir / Path("slurm_out.rst7")]
            if all([path.exists() for path in output_paths]):
                self.bss_systems["free"][name] = [str(path) for path in output_paths]
            else:
                self.failed_isomers[name] = {
                    "system": self.bss_systems["free"][name],
                    "reason": f"Failed to parameterise {name}.",
                }
                del self.bss_systems["free"][name]

        # Check in case all isomers failed and we have nothing left in bss_systems
        if not self.bss_systems["free"]:
            raise ValueError("All isomers failed to parameterise.")

        # Parameterise the protein (and any waters). No need to use run_command as this is cheap.
        self.logger.info("Parameterising protein...")
        protein_path = self.inp_protein.receive()
        prot_param_dir = self.work_dir / Path("parameterise_protein")
        protein_sys = BSS.IO.readMolecules(str(protein_path))
        protein_molecules = protein_sys.getMolecules()
        # Parameterise molecules (protein chains or waters) individually
        param_mol = []
        for molecule in protein_molecules:
            param_mol.append(
                BSS.Parameters.parameterise(
                    molecule,
                    forcefield=self.protein_force_field.value,
                    water_model=self.water_model.value,
                    work_dir=str(prot_param_dir),
                )
                .getMolecule()
                .toSystem()
            )
        protein_system = param_mol[0]
        for mol in param_mol[1:]:
            protein_system += mol

        # Save the protein system and set all the bound systems to point to this path
        BSS.IO.saveMolecules(
            str(prot_param_dir / Path("protein.prm7")), protein_system, ["prm7", "rst7"]
        )
        protein_paths = [
            str(prot_param_dir / Path("protein.prm7")),
            str(prot_param_dir / Path("protein.rst7")),
        ]

        for name in self.bss_systems["free"]:
            self.bss_systems["bound"][name] = protein_paths

        # Delete the system to save memory
        del protein_system

    def _set_up_cofactor(self) -> None:
        """Parameterise the cofactor and add it to the protein."""
        # This is compute intensive, so write out a BSS script,
        # run it with run_command, then read back in the output
        import BioSimSpace as BSS

        if self.inp_cofactor_sdf.is_connected:
            self.logger.info("Parameterising cofactor...")

            # Create a python script to be run through run_command (so that we use the
            # desired scheduling system)
            work_dir = self.work_dir / Path("parameterise_cofactor")
            work_dir.mkdir(parents=True, exist_ok=True)
            param_script = (
                "import BioSimSpace as BSS\n"
                "from maize.steps.exs.biosimspace._utils import rename_lig\n"
                f"system = BSS.IO.readMolecules('{self.inp_cofactor_sdf.receive()}')\n"
                "system = system[0]\n"
                "param_mol = BSS.Parameters.parameterise(\n"
                "    system,\n"
                f"    forcefield='{self.ligand_force_field.value}',\n"
                f"    work_dir='{work_dir}',\n"
                ").getMolecule().toSystem()\n"
                "rename_lig(param_mol, new_name='COF')\n"
                f'BSS.IO.saveMolecules("{work_dir}/slurm_out", param_mol, ["prm7", "rst7"])\n'
            )

            # Write the script to a file
            with open(work_dir / Path("param_script.py"), "w") as f:
                f.write(param_script)

            # Run the script
            self.run_command(f"python {work_dir / Path('param_script.py')}")

            # Check that the output has been successfully generated. If so,
            # load it back in and add it to the parameterised protein system
            output_paths = [work_dir / Path("slurm_out.prm7"), work_dir / Path("slurm_out.rst7")]
            if all([path.exists() for path in output_paths]):
                cofactor_system = BSS.IO.readMolecules([str(path) for path in output_paths])
                protein_system = BSS.IO.readMolecules(
                    [str(path) for path in self.bss_systems["bound"]]
                )
                protein_system += cofactor_system
                BSS.IO.saveMolecules(
                    str(work_dir / Path("protein_cofactor.prm7")), protein_system, ["prm7", "rst7"]
                )
                protein_paths = [
                    str(work_dir / Path("protein_cofactor.prm7")),
                    str(work_dir / Path("protein_cofactor.rst7")),
                ]

                for name in self.bss_systems["free"]:
                    self.bss_systems["bound"][name] = protein_paths

                # Delete the system to save memory
                del protein_system

            else:
                self.logger.error(f"Failed to parameterise the cofactor. Please check {work_dir}.")

    def _assemble_complexes(self) -> None:
        """Combine the parameterised protein and ligands to create the complexes."""
        import BioSimSpace as BSS

        self.logger.info("Assembling complexes...")
        for name in self.bss_systems["free"]:
            protein = BSS.IO.readMolecules([str(path) for path in self.bss_systems["bound"][name]])
            ligand = BSS.IO.readMolecules([str(path) for path in self.bss_systems["free"][name]])
            complx = ligand + protein
            save_dir = self.work_dir / Path("complexes") / Path(name)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_files = [save_dir / Path("complx.prm7"), save_dir / Path("complx.rst7")]
            BSS.IO.saveMolecules(str(save_dir / Path("complx")), complx, ["prm7", "rst7"])
            self.bss_systems["bound"][name] = save_files
            del complx, protein, ligand

    def _solvate(self) -> None:
        """Solvate the systems and add the desired salt concentration."""

        self.logger.info("Solvating systems...")
        cmds = []
        output_dirs = {}
        for leg in self.bss_systems:
            output_dirs[leg] = {}
            for name, system_paths in self.bss_systems[leg].items():
                work_dir = self.work_dir / Path("solvate") / Path(leg) / Path(name)
                work_dir.mkdir(parents=True, exist_ok=True)
                output_dirs[leg][name] = work_dir

                # Create a python script to be run through run_command (so that we use the
                # desired scheduling system)
                script = (
                    "import BioSimSpace as BSS\n"
                    f"system = BSS.IO.readMolecules({[str(path) for path in system_paths]})\n"
                    "non_waters = [mol for mol in system if mol.nAtoms() != 3]\n"
                    "dry_system = BSS._SireWrappers._system.System(non_waters)\n"
                    "box_min, box_max = dry_system.getAxisAlignedBoundingBox()\n"
                    "box_size = [y - x for x, y in zip(box_min, box_max)]\n"
                    "padding = 12 * BSS.Units.Length.angstrom\n"
                    "box_length = max(box_size) + 2 * padding\n"
                    "box, angles = BSS.Box.cubic(box_length)\n"
                    "try:\n"
                    "    waters_to_exclude = [wat for wat in system.search(f'water and not (water within {padding} of protein)').molecules()]\n"
                    "    if len(waters_to_exclude) > 0:\n"
                    "        print(f'Excluding {len(waters_to_exclude)} waters that are over {padding} A from the protein')\n"
                    "except ValueError:\n"
                    "    waters_to_exclude = []\n"
                    "system.removeMolecules(waters_to_exclude)\n"
                    "solvated_system = BSS.Solvent.solvate(\n"
                    f"    model='{self.water_model.value}',\n"
                    "    molecule=system,\n"
                    "    box=box,\n"
                    "    angles=angles,\n"
                    f"    ion_conc={self.ion_conc.value},\n"
                    ")\n"
                    f"BSS.IO.saveMolecules('{work_dir / Path('solvated')}', solvated_system, ['prm7', 'rst7'])\n"
                )

                # Write the script to a file
                with open(work_dir / Path("solvate_script.py"), "w") as f:
                    f.write(script)

                cmds.append(f"python {work_dir / Path('solvate_script.py')}")

        # Run for all ligands in parallel
        self.logger.info("Solvating all systems...")
        options = JobResourceConfig(custom_attributes={"mem": "24GB"})
        self.run_multi(cmds, batch_options=options)

        # Collect all the results and update the bss_systems dictionary. If one has failed, log this
        # and add it to the failed dictionary.
        for leg, leg_dirs in output_dirs.items():
            for name, work_dir in leg_dirs.items():
                # Check if we have the expected output files
                output_files = [
                    work_dir / Path("solvated.prm7"),
                    work_dir / Path("solvated.rst7"),
                ]
                if all([path.exists() for path in output_files]):
                    self.bss_systems[leg][name] = [str(path) for path in output_files]
                else:
                    self.failed_isomers[name] = {
                        "system": self.bss_systems[leg][name],
                        "reason": f"Failed to solvate {name}.",
                    }
                    del self.bss_systems[leg][name]

        # Check in case all isomers failed and we have nothing left in bss_systems
        if not any(self.bss_systems.values()):
            raise ValueError("All isomers failed to solvate.")

    def _minimise(self) -> None:
        """Minimise the systems."""

        self.logger.info("Minimising systems...")

        # Create a protocol for minimisation
        protocol_str = "protocol = BSS.Protocol.Minimisation(steps=10_000)\n"

        # Run minimisation for both legs
        for leg in self.bss_systems:
            self._run_protocol(protocol_str, "bound", "minimisation")

    def _equil_nvt_restrained(self) -> None:
        """Equilibrate the systems in NVT with restraints."""

        self.logger.info("Equilibrating systems in NVT with restraints...")
        protocol_str = (
            "protocol = BSS.Protocol.Equilibration("
            "runtime=5 * BSS.Units.Time.picosecond, "
            f"temperature_start=0 * BSS.Units.Temperature.kelvin, "
            f"temperature_end={self.temperature.value} * BSS.Units.Temperature.kelvin, "
            "restraint='all')\n"
        )
        for leg in self.bss_systems:
            self._run_protocol(protocol_str, leg, "equil_nvt_restrained")

    def _equil_nvt_backbone_restrained(self) -> None:
        """Equilibrate the complex in NVT with restraints on the backbone."""

        self.logger.info("Equilibrating complexes in NVT with restraints on the backbone...")

        protocol_str = (
            "protocol = BSS.Protocol.Equilibration("
            "runtime=5 * BSS.Units.Time.picosecond, "
            f"temperature={self.temperature.value} * BSS.Units.Temperature.kelvin, "
            "restraint='backbone')\n"
        )
        self._run_protocol(protocol_str, "bound", "equil_short_nvt_backbone_restrained")

    def _equil_nvt_unrestrained(self) -> None:
        """Equilibrate the systems in NVT without restraints."""

        self.logger.info("Equilibrating systems in NVT without restraints...")

        protocol_str = (
            "protocol = BSS.Protocol.Equilibration("
            f"runtime=50 * BSS.Units.Time.picosecond, "
            f"temperature={self.temperature.value} * BSS.Units.Temperature.kelvin)\n"
        )

        for leg in self.bss_systems:
            self._run_protocol(protocol_str, leg, "equil_short_nvt")

    def _equil_npt_restrained(self) -> None:
        """Equilibrate the systems in NPT with restraints on non-solvent heavy atoms."""

        self.logger.info(
            "Equilibrating systems in NPT with restraints on non-solvent heavy atoms..."
        )
        run_time_ps = 50 if self.equilibration_protocol.value == "fast" else 400

        protocol_str = (
            # CHANGEME
            "protocol = BSS.Protocol.Equilibration("
            f"runtime={run_time_ps} * BSS.Units.Time.picosecond, "
            f"temperature={self.temperature.value} * BSS.Units.Temperature.kelvin, "
            "pressure=1 * BSS.Units.Pressure.atm, "
            "restraint='heavy')\n"
        )

        for leg in self.bss_systems:
            self._run_protocol(protocol_str, leg, "equil_short_npt")

    def _equil_npt_unrestrained(self) -> None:
        """Equilibrate the systems in NPT without restraints."""

        self.logger.info("Equilibrating systems in NPT without restraints...")
        run_time_ps = 50 if self.equilibration_protocol.value == "fast" else 1000

        protocol_str = (
            "protocol = BSS.Protocol.Equilibration("
            f"runtime={run_time_ps} * BSS.Units.Time.picosecond, "
            f"temperature={self.temperature.value} * BSS.Units.Temperature.kelvin, "
            "pressure=1 * BSS.Units.Pressure.atm)\n"
        )

        for leg in self.bss_systems:
            self._run_protocol(protocol_str, leg, "equil_short_npt")

    def _run_protocol(
        self,
        protocol_code: str,
        leg: Literal["bound", "free"],
        protocol_name: str,
    ) -> None:
        """
        Run a standard protocol using GROMACS. The parameterisation
        and free energy stages cannot be run with this function. Output files
        are saved to a protocol_name / leg / isomer directory as "bss_system.prm7" and
        "bss_system.rst7".
        """
        # Write python scripts for each isomer, using the protocol code provided
        cmds = []
        output_dirs = {}
        for name, system_paths in self.bss_systems[leg].items():
            work_dir = self.work_dir / Path(protocol_name) / Path(leg) / Path(name)
            work_dir.mkdir(parents=True, exist_ok=True)
            output_dirs[name] = work_dir

            # Create a python script to be run through run_command (so that we use the
            # desired scheduling system)
            script = (
                "import BioSimSpace as BSS\n"
                f"system = BSS.IO.readMolecules({[str(path) for path in system_paths]})\n"
                f"{protocol_code}"
                f"process = BSS.Process.Gromacs(system, protocol=protocol, work_dir='{work_dir}')\n"
                "process.addArgs({'-ntmpi': '1'})\n"
                "process.start()\n"
                "out_system = process.getSystem(block=True)\n"
                f"BSS.IO.saveMolecules('{work_dir / Path('bss_system')}', out_system, ['prm7', 'rst7'], property_map={{'velocity': 'foo'}})\n"
            )

            # Write the script to a file
            with open(work_dir / Path("bss_script.py"), "w") as f:
                f.write(script)

            cmds.append(f"python {work_dir / Path('bss_script.py')}")

        # Run for all ligands in parallel
        self.logger.info(f"Running {protocol_name} for the {leg} leg for all isomers...")
        options = JobResourceConfig(custom_attributes={"gres": "gpu:1", "mem": "24GB"})
        self.run_multi(cmds, batch_options=options)

        # Collect all the results and update the bss_systems dictionary. If one has failed, log this
        # and add it to the failed dictionary.
        for name, work_dir in output_dirs.items():
            # Check if we have the expected output files
            output_files = [
                work_dir / Path("bss_system.prm7"),
                work_dir / Path("bss_system.rst7"),
            ]
            if all([path.exists() for path in output_files]):
                self.bss_systems[leg][name] = output_files
            else:
                self.logger.error(f"Failed to run {protocol_name} for {name} for {leg} leg.")
                self.failed_isomers[name] = {
                    "system": self.bss_systems[leg][name],
                    "reason": f"Failed to run {protocol_name} for {name} for {leg} leg.",
                }
                # Remove from the bss_systems dictionary
                del self.bss_systems[leg][name]

        # Check in case all isomers failed and we have nothing left in bss_systems
        if not self.bss_systems[leg]:
            raise ValueError(f"All isomers failed to run {protocol_name} for the {leg} leg.")

    def _ensemble_equilibration(self) -> None:
        """Run production MD for each repeat to generate diverse starting structures."""
        # As before, do this by writing python files to be run with run_command, but this time
        # store the paths to the pre-equilibrated systems dict
        self.logger.info("Running ensemble equilibration...")
        cmds = []
        output_dirs = {}

        for leg in self.bss_systems:
            output_dirs[leg] = {}
            for name, system_paths in self.bss_systems[leg].items():
                output_dirs[leg][name] = []
                for repeat_no in range(1, self.n_repeats.value + 1):
                    work_dir = (
                        self.work_dir
                        / Path("ensemble_equil")
                        / Path(leg)
                        / Path(name)
                        / Path(f"repeat_{repeat_no}")
                    )
                    work_dir.mkdir(parents=True, exist_ok=True)
                    output_dirs[leg][name].append(work_dir)

                    # Create a python script to be run through run_command (so that we use the
                    # desired scheduling system)
                    script = (
                        "import BioSimSpace as BSS\n"
                        f"system = BSS.IO.readMolecules({[str(path) for path in system_paths]})\n"
                        "protocol = BSS.Protocol.Equilibration("
                        f"runtime={self.run_time_ensemble_equilibration.value} * BSS.Units.Time.nanosecond, "
                        f"temperature={self.temperature.value} * BSS.Units.Temperature.kelvin, "
                        "pressure=1 * BSS.Units.Pressure.atm)\n"
                        f"process = BSS.Process.Gromacs(system, protocol=protocol, work_dir='{work_dir}')\n"
                        "process.addArgs({'-ntmpi': '1'})\n"
                        "process.start()\n"
                        "out_system = process.getSystem(block=True)\n"
                        f"BSS.IO.saveMolecules('{work_dir / Path('bss_system')}', out_system, ['prm7', 'rst7'], property_map={{'velocity': 'foo'}})\n"
                    )

                    # Write the script to a file
                    with open(work_dir / Path("bss_script.py"), "w") as f:
                        f.write(script)

                    cmds.append(f"python {work_dir / Path('bss_script.py')}")

        # Run for all ligands in parallel
        options = JobResourceConfig(custom_attributes={"gres": "gpu:1", "mem": "24GB"})
        self.run_multi(cmds, batch_options=options)

        # Collect all the results and update the bss_systems dictionary. If one has failed, log this
        # and add it to the failed dictionary.
        for leg in output_dirs:
            for name in output_dirs[leg]:
                self._preequilibrated_systems[leg][name] = []
                for work_dir in output_dirs[leg][name]:
                    # Check if we have the expected output files
                    output_files = [
                        work_dir / Path("bss_system.prm7"),
                        work_dir / Path("bss_system.rst7"),
                    ]
                    if all([path.exists() for path in output_files]):
                        self._preequilibrated_systems[leg][name].append(output_files)
                    else:
                        self._logger.error(
                            f"Failed to run ensemble equilibration for {name} for {leg} leg."
                        )
                        self.failed_isomers[name] = {
                            "system": self.bss_systems[leg][name],
                            "reason": f"Failed to run ensemble equilibration for {name} for {leg} leg.",
                        }
                        # Remove from the bss_systems dictionary
                        del self.bss_systems[leg][name]
                        break

        # Check in case all isomers failed and we have nothing left in bss_systems
        if not self.bss_systems[leg]:
            raise ValueError("All isomers failed to run ensemble equilibration.")

    def _set_up_abfe(self) -> None:
        """Generate the restraints and prepare the ABFE run directories."""
        # Map stages to perturbation types
        pert_type_map = {
            "restrain": "restraint",
            "discharge": "discharge_soft",
            "vanish": "vanish_soft",
        }

        # Define code to get extra options for SOMD
        extra_options_code = (
            # Work out a reasonable number of cycles as the default is terrible (
            # produces far too many cycles). Once every 25000 steps is a good
            # starting point.
            "total_steps = round(\n"
            f"    {self.run_time_abfe.value} * 1e6 / {self.production_timestep.value}\n"
            ")  # runtime in ns, timestep in fs\n"
            "if total_steps < 25000:  # Very short run - just run one cycle\n"
            "    ncycles = 1\n"
            "    nmoves = total_steps\n"
            "else:\n"
            "    ncycles = total_steps // 25000\n"
            "    nmoves = 25000\n"
            "if ncycles * nmoves < total_steps:\n"
            "    self.logger.warning(\n"
            f"        f'The number of steps is not divisible by 25000. The rumtime will be {{ncycles * nmoves * {self.production_timestep.value} / 1e6}} ns'\n"
            f"        f' rather than {self.run_time_abfe.value} ns.'\n"
            "    )\n"
            "extra_options = {\n"
            "    'ncycles': ncycles,\n"
            "    'nmoves': nmoves,\n"
            "    'hydrogen mass repartitioning factor': 3.0,\n"
            "    'cutoff distance': '12 * angstrom',  # As we use RF\n"
            "    'integrator': 'langevinmiddle',\n"
            "    'inverse friction': '1 * picosecond',\n"
            "    'thermostat': False,  # Handled by langevin integrator\n"
            "    'minimise': True,\n"
            "}\n"
        )

        # As usual, do this by creating python scripts to be run with run_multi
        self.logger.info("Setting up ABFE calculations...")
        cmds = []

        for leg in self.bss_systems:
            stages = (
                ["restrain", "discharge", "vanish"] if leg == "bound" else ["discharge", "vanish"]
            )
            for name, system_paths in self.bss_systems[leg].items():
                self._abfe_dirs[leg][name] = {}
                for stage in stages:
                    self._abfe_dirs[leg][name][stage] = []
                    for repeat_no in range(1, self.n_repeats.value + 1):
                        work_dir = (
                            self.work_dir
                            / Path("abfe")
                            / Path(leg)
                            / Path(name)
                            / Path(f"repeat_{repeat_no}")
                            / Path(stage)
                        )
                        work_dir.mkdir(parents=True, exist_ok=True)
                        self._abfe_dirs[leg][name][stage].append(work_dir)

                        # Create a python script to be run through run_command (so that we use the
                        # desired scheduling system)
                        if leg == "free":
                            script = (
                                "import BioSimSpace.Sandpit.Exscientia as BSS\n"
                                "from maize.steps.exs.biosimspace._utils import mark_ligand_for_decoupling\n"
                                f"system = BSS.IO.readMolecules({[str(path) for path in self._preequilibrated_systems[leg][name][repeat_no - 1]]})\n"
                                "mark_ligand_for_decoupling(system, ligand_name='LIG')\n"
                                "protocol = BSS.Protocol.FreeEnergy(\n"
                                f"    runtime={self.run_time_abfe.value} * BSS.Units.Time.nanosecond,\n"
                                f"    timestep={self.production_timestep.value} * BSS.Units.Time.femtosecond,\n"
                                f"    temperature={self.temperature.value} * BSS.Units.Temperature.kelvin,\n"
                                f"    lam_vals={self.lambda_schedule_free.value[stage] if leg == 'free' else self.lambda_schedule_bound.value[stage]},\n"
                                f"    perturbation_type='{pert_type_map[stage]}',\n"
                                ")\n"
                                f"{extra_options_code}"
                                f"process = BSS.FreeEnergy.AlchemicalFreeEnergy(system, protocol=protocol, engine='somd',extra_options=extra_options, work_dir='{work_dir}', setup_only=True)\n"
                            )
                        else:  # Need to get the restraint for the bound leg
                            restraint_search_dir = self._preequilibrated_systems[leg][name][
                                repeat_no - 1
                            ][0].parent
                            restraint_traj_path = str(restraint_search_dir / Path("gromacs.xtc"))
                            restraint_top_path = str(restraint_search_dir / Path("gromacs.tpr"))
                            script = (
                                "import BioSimSpace.Sandpit.Exscientia as BSS\n"
                                "from maize.steps.exs.biosimspace._utils import mark_ligand_for_decoupling\n"
                                f"system = BSS.IO.readMolecules({[str(path) for path in self._preequilibrated_systems[leg][name][repeat_no - 1]]})\n"
                                "mark_ligand_for_decoupling(system, ligand_name='LIG')\n"
                                "traj = BSS.Trajectory.Trajectory(\n"
                                f"    trajectory='{restraint_traj_path}',\n"
                                f"    topology='{restraint_top_path}',\n"
                                f"    system=system,\n"
                                ")\n"
                                "# Search for the optimal Boresch restraints\n"
                                "restraint = BSS.FreeEnergy.RestraintSearch.analyse(\n"
                                f"    work_dir = '{restraint_search_dir}',\n"
                                "    system = system,\n"
                                "    traj = traj,\n"
                                "    temperature = 298 * BSS.Units.Temperature.kelvin,\n"
                                "    method='BSS',\n"
                                "    restraint_type = 'Boresch',\n"
                                ")\n"
                                "restraint_corr = restraint.getCorrection().value()\n"
                                f"with open('{work_dir}/restraint_corr.txt', 'w') as f:\n"
                                "    f.write(f'{restraint_corr}')\n"
                                "protocol = BSS.Protocol.FreeEnergy(\n"
                                f"    runtime={self.run_time_abfe.value} * BSS.Units.Time.nanosecond,\n"
                                f"    timestep={self.production_timestep.value} * BSS.Units.Time.femtosecond,\n"
                                f"    temperature={self.temperature.value} * BSS.Units.Temperature.kelvin,\n"
                                f"    lam_vals={self.lambda_schedule_free.value[stage] if leg == 'free' else self.lambda_schedule_bound.value[stage]},\n"
                                f"    perturbation_type='{pert_type_map[stage]}',\n"
                                ")\n"
                                f"{extra_options_code}"
                                f"process = BSS.FreeEnergy.AlchemicalFreeEnergy(system, protocol=protocol, restraint=restraint, engine='somd', extra_options=extra_options, work_dir='{work_dir}', setup_only=True)\n"
                            )

                        # Write the script to a file
                        with open(work_dir / Path("bss_script.py"), "w") as f:
                            f.write(script)

                        cmds.append(f"python {work_dir / Path('bss_script.py')}")

        # Run for all ligands in parallel
        options = JobResourceConfig(custom_attributes={"mem": "24GB"})
        self.run_multi(cmds, batch_options=options)

    def _run_abfe(self) -> None:
        """Run the absolute ABFE calculations."""

        self.logger.info("Running ABFE calculations...")
        cmds = []

        # Collect all of the lambda directories for all repeats...
        for leg in self._abfe_dirs:
            stages = (
                ["restrain", "discharge", "vanish"] if leg == "bound" else ["discharge", "vanish"]
            )
            for name in self._abfe_dirs[leg]:
                for repeat_no in range(1, self.n_repeats.value + 1):
                    for stage in stages:
                        work_dir = self._abfe_dirs[leg][name][stage][repeat_no - 1]
                        for lam_dir in [d for d in work_dir.iterdir() if d.is_dir()]:
                            # Write a simple bash script to change to the directory and run the ABFE
                            # calculation
                            script = f"cd {lam_dir}\nsomd-freenrg -C somd.cfg -t somd.prm7 -c somd.rst7 -m somd.pert\n"
                            with open(lam_dir / Path("run_script.sh"), "w") as f:
                                f.write(script)

                            cmds.append(f"bash {lam_dir / Path('run_script.sh')}")

        # Run for all ligands in parallel
        options = JobResourceConfig(custom_attributes={"gres": "gpu:1", "mem": "24GB"})
        self.run_multi(cmds, batch_options=options)

    def _analyse_abfe(self) -> None:
        """Analyse the ABFE results."""

        self.logger.info("Analysing ABFE results...")
        cmds = []

        # Create jobs to run all of the mbar analyses
        for leg in self._abfe_dirs:
            for name in self._abfe_dirs[leg]:
                for repeat_no in range(1, self.n_repeats.value + 1):
                    for stage in self._abfe_dirs[leg][name]:
                        work_dir = self._abfe_dirs[leg][name][stage][repeat_no - 1]

                        script = (
                            "import BioSimSpace.Sandpit.Exscientia as BSS\n"
                            f"pmf, _ = BSS.FreeEnergy.AlchemicalFreeEnergy.analyse('{work_dir}', estimator='MBAR')\n"
                            "dg = pmf[-1][1].value()\n"
                            "er= pmf[-1][2].value()\n"
                            f"with open('{work_dir}/dg.txt', 'w') as f:\n"
                            "    f.write(f'{dg},{er}')\n"
                        )

                        with open(work_dir / Path("analysis_script.py"), "w") as f:
                            f.write(script)

                        cmds.append(f"python {work_dir / Path('analysis_script.py')}")

        # Run for all ligands in parallel
        options = JobResourceConfig(custom_attributes={"mem": "24GB"})
        self.run_multi(cmds, batch_options=options)

        # Read in all of the results
        results = {
            name: {"dg": np.zeros(self.n_repeats.value), "er": np.zeros(self.n_repeats.value)}
            for name in self._abfe_dirs["free"]
        }
        for leg in self._abfe_dirs:
            dg_multiplier = 1 if leg == "free" else -1
            for name in self._abfe_dirs[leg]:

                # Get all the results for the calculations
                for repeat_no in range(1, self.n_repeats.value + 1):
                    for stage in self._abfe_dirs[leg][name]:
                        work_dir = self._abfe_dirs[leg][name][stage][repeat_no - 1]
                        if (work_dir / Path("dg.txt")).exists:
                            with open(work_dir / Path("dg.txt"), "r") as f:
                                dg, er = f.read().split(",")

                                # Check for crazy restraining free energies
                                if stage == "restrain":
                                    if float(dg) > 3:
                                        self.logger.warning(
                                            f"The free energy of turning on the restraints for {name} run {repeat_no} is"
                                            f" greater than 3 kcal/mol ({dg} kcal/mol). This may suggest a change in binding"
                                            " mode or a overly short restraints fit from a very short simulation."
                                        )

                                # Store the free energy change
                                results[name]["dg"][repeat_no - 1] += dg_multiplier * float(dg)
                                # Add errors in quadrature
                                results[name]["er"][repeat_no - 1] += np.sqrt(
                                    np.sum(
                                        np.array([results[name]["er"][repeat_no - 1], float(er)])
                                        ** 2
                                    )
                                )
                        else:
                            self.logger.error(
                                f"Failed to analyse ABFE results for {name} for {leg} leg."
                            )
                            self.failed_isomers[name] = {
                                "system": self.bss_systems[leg][name],
                                "reason": f"Failed to analyse ABFE results for {name} for {leg} leg.",
                            }
                            # Add np.nan to the results
                            results[name]["dg"][repeat_no - 1] = np.nan
                            results[name]["er"][repeat_no - 1] = np.nan

                    # If this is the bound leg, add the restraint correction
                    if leg == "bound":
                        restraint_corr = 0
                        work_dir = self._abfe_dirs[leg][name]["restrain"][repeat_no - 1]
                        with open(work_dir / Path("restraint_corr.txt"), "r") as f:
                            restraint_corr += float(f.read())
                        results[name]["dg"][repeat_no - 1] -= restraint_corr

        # Convert to ABFE result objects and save to the instance variable
        smiles = {isomer.inchi: isomer.to_smiles() for isomer in self.isomers}
        self.results = {
            name: ABFEResult(
                smiles=smiles[name],
                inchi_key=name,
                dg_array=results[name]["dg"],
                error_array=results[name]["er"],
            )
            for name in results
        }

    def _save_results_file(self) -> None:
        """Save the results to a file."""
        results_file = self.work_dir / Path("results.json")
        self.logger.info(f"Results are {self.results}")
        self.logger.info(f"Saving results to {results_file}...")
        with open(results_file, "w") as f:
            json.dump(self.results, f, cls=ABFEResultEncoder)

    def _dump_data(self) -> None:
        """Dump all the data to the dump directory."""
        import shutil

        if self.dump_to:
            dump_dir = self.dump_to.value

            # If it exists, move the old dump directory
            i = 0
            while dump_dir.exists():
                dump_dir = self.dump_to.value.parent / Path(f"dump_{i}")
                i += 1

            dump_dir.mkdir(parents=True, exist_ok=True)

            # Simply copy over everything in the work_dir
            for item in self.work_dir.iterdir():
                # Copy even if it is a directory
                if item.is_dir():
                    shutil.copytree(item, dump_dir / item.name)
                else:
                    shutil.copy(item, dump_dir)


@pytest.fixture
def multi_sdf_path(shared_datadir: Path) -> Path:
    return shared_datadir / "2_benzenes.sdf"


@pytest.fixture
def t4l_protein_pdb_path(shared_datadir: Path) -> Path:
    return shared_datadir / "t4l.pdb"


class TestSuiteABFE:
    def test_biosimspace_abfe(
        self,
        temp_working_dir: Any,
        multi_sdf_path: Path,
        t4l_protein_pdb_path: Path,
    ) -> None:
        """Test the BioSimSpace ABFE node."""

        rig = TestRig(BssAbfe)
        dump_dir = Path().absolute().parents[1] / "dump"
        # Load the isomers
        # Load the isomers
        isomer_collections = load_sdf_library(multi_sdf_path, split_strategy="none")

        # Convert to lists of isomers since we've read in all the sdfs individually
        isomers = [
            isomer
            for isomer_collection in isomer_collections
            for isomer in isomer_collection.molecules
        ]
        res = rig.setup_run(
            inputs={"inp": [isomers], "inp_protein": [t4l_protein_pdb_path]},
            parameters={
                "temperature": 298.15,
                "protein_force_field": "ff14SB",
                "ligand_force_field": "openff_unconstrained-2.0.0",
                "water_model": "tip3p",
                "ion_conc": 0.15,
                "equilibration_protocol": "fast",
                "run_time_abfe": 0.1,
                "n_repeats": 3,
                # Make a simple lambda schedule to run quickly
                # "lambda_schedule_free": {
                #     "discharge": [0.0, 1.0],
                #     "vanish": [0.0, 1.0],
                # },
                # "lambda_schedule_bound": {
                #     "restrain": [0.0, 1.0],
                #     "discharge": [0.0, 1.0],
                #     "vanish": [0.0, 1.0],
                # },
                "n_jobs": 500,
                "production_timestep": 4.0,
                "dump_to": dump_dir,
            },
        )
        output = res["out"].get()

        # Check that results have been returned
        # Check that output["UHOVQNZJYSORNB-UHFFFAOYNA-N"] is an ABFEResult object
        assert isinstance(output["UHOVQNZJYSORNB-UHFFFAOYNA-N"], ABFEResult)
        abfe_res = output["UHOVQNZJYSORNB-UHFFFAOYNA-N"]
        assert abfe_res.smiles == "[H]c1c([H])c([H])c([H])c([H])c1[H]"
        assert abfe_res.inchi_key == "UHOVQNZJYSORNB-UHFFFAOYNA-N"
        assert abfe_res.dg_array.shape == (2,)
        assert isinstance(abfe_res.dg_array, np.ndarray)
        assert abfe_res.error_array.shape == (2,)
        assert isinstance(abfe_res.error_array, np.ndarray)
        assert isinstance(abfe_res.dg, float)
        assert isinstance(abfe_res.ci_95, float)

        # Check that the dumping worked
        dump_output_dir = sorted(dump_dir.iterdir())[-1]
        assert dump_output_dir.exists()
