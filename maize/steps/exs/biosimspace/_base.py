"""A BioSimSpace base Node for running BioSimSpace simulations."""

# pylint: disable=import-outside-toplevel, import-error
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path

from maize.core.interface import FileParameter, Input, Output
from maize.core.node import Node

from ._utils import _ClassProperty
from .enums import _ENGINE_CALLABLES, BSSEngine
from .exceptions import BioSimSpaceNullSystemError

__all__ = []


class _BioSimSpaceBase(Node, ABC):
    """
    Abstract base node for all BioSimSpace nodes. All non-abstract
    subclasses should set the biosimspace_engine attribute,
    and the required_callables attribute will be set automatically.
    They should also implment the run and _get_process methods.

    Notes
    -----
    Install with `mamba create -f env.yaml`.

    References
    ----------
    L. O. Hedges et al., JOSS, 2019, 4, 1831.
    L. O. Hedges et al., LiveCoMS, 2023, 5, 2375–2375.
    """

    required_packages = ["BioSimSpace"]

    bss_engine: BSSEngine
    """The engine to use in the subclass."""

    @_ClassProperty
    def required_callables(cls) -> list[str]:
        return _ENGINE_CALLABLES[cls.bss_engine]

    # Input
    inp: Input[Path | list[Path]] = Input()
    """
    Path(s) to system input files. These can be in any of the formats
    given by BSS.IO.fileFormats():
    
    gro87, grotop, mol2, pdb, pdbx, prm7, rst rst7, psf, sdf

    And can also be perturbable systems with the .bss extension.
    """

    # Parameters
    dump_to: FileParameter[Path] = FileParameter(optional=True)
    """A folder to dump all generated data to"""

    # Output
    out: Output[list[Path]] = Output(mode="copy")
    """The output files generated by BioSimSpace, in gromacs format."""

    @abstractmethod
    def run(self) -> None:
        """All subclasses must implement a run method."""

    def _get_protocol(self) -> "BSS.Protocol._protocol.Protocol | None":
        """Get the process to run. This should only be called within `run()`."""
        raise NotImplementedError

    def _load_input(self, sandpit: bool = False) -> "BioSimSpace._SireWrappers.System":
        """Load the input files. This should only be called within `run()`."""

        if sandpit:
            import BioSimSpace.Sandpit.Exscientia as BSS
        else:
            import BioSimSpace as BSS

        input_files = self.inp.receive()

        # Convert to strings for BSS
        input_files = (
            [str(f) for f in input_files] if isinstance(input_files, list) else str(input_files)
        )

        return BSS.IO.readMolecules(input_files)

    def _load_perturbable_input(self) -> "BioSimSpace._SireWrappers.System":
        """Load the perturbable system input files. This should only be called within `run()`."""

        import BioSimSpace.Sandpit.Exscientia as BSS

        input_files = self.inp.receive()

        # Make sure that we have all the required files to load a perturbable system
        required_file_endings = {
            "top0": "0.prm7",
            "coords0": "0.rst7",
            "top1": "1.prm7",
            "coords1": "1.rst7",
        }

        # Get the files
        files = {k: None for k in required_file_endings.keys()}
        for k, v in required_file_endings.items():
            for f in input_files:
                if f.name.endswith(v):
                    files[k] = f
                    break

        # Check that we have all the files
        if None in files.values():
            raise ValueError(
                f"Could not find files with all required endings: {required_file_endings.values()}"
            )

        # Convert to strings for BSS
        files = {k: str(v) for k, v in files.items()}

        return BSS.IO.readPerturbableSystem(
            top0=files["top0"],
            coords0=files["coords0"],
            top1=files["top1"],
            coords1=files["coords1"],
        )

    def _save_output(
        self, system: "BioSimSpace._SireWrappers.System", sandpit: bool = False
    ) -> None:
        """Save the output files. This should only be called within `run()`."""
        if sandpit:
            import BioSimSpace.Sandpit.Exscientia as BSS
        else:
            import BioSimSpace as BSS

        self.out.send(
            [Path(f) for f in BSS.IO.saveMolecules("bss_system", system, ["gro87", "grotop"])]
        )

    def _save_pertrubable_output(self, system: "BioSimSpace._SireWrappers.System") -> None:
        """Save the perturbable output files. This should only be called within `run()`."""
        import BioSimSpace.Sandpit.Exscientia as BSS

        file_base = "bss_system"
        BSS.IO.savePerturbableSystem(file_base, system)
        self.out.send(
            [
                Path(f)
                for f in [
                    f"{file_base}{end}.{ext}" for end in ["0", "1"] for ext in ["prm7", "rst7"]
                ]
            ]
        )

    def _run_process(self) -> None:
        """
        Run the BioSimSpace process returned by `_get_process`.
        using the engine specified by `bss_engine`. This should
        only be called within `run()`.
        """
        import BioSimSpace as BSS

        # Map the BSS Engines to process classes
        process_map = {
            BSSEngine.GROMACS: BSS.Process.Gromacs,
            BSSEngine.SANDER: BSS.Process.Amber,
            BSSEngine.PMEMD: BSS.Process.Amber,
            BSSEngine.PMEMD_CUDA: BSS.Process.Amber,
            BSSEngine.OPENMM: BSS.Process.OpenMM,
            BSSEngine.SOMD: BSS.Process.Somd,
            BSSEngine.NAMD: BSS.Process.Namd,
        }

        # Get the input
        system = self._load_input()

        # Get the protocol
        protocol = self._get_protocol()

        # Get the process
        process = process_map[self.bss_engine](
            system,
            protocol=protocol,
            exe=self._get_executable(),
            work_dir=str(self.work_dir),
        )

        # Run the process and wait for it to finish
        self.logger.info(f"Running {protocol.__class__.__name__} with {self.bss_engine.name}...")
        cmd = " ".join([process.exe(), process.getArgString()])
        self.run_command(cmd)
        output_system = process.getSystem(block=True)
        # BioSimSpace sometimes returns None, so we need to check
        if output_system is None:
            raise BioSimSpaceNullSystemError("The output system is None.")

        # Save the output
        self._save_output(output_system)

        # Dump the data
        self._dump_data()

    def _dump_data(self) -> None:
        """Dump the data to the dump_to folder."""
        if self.dump_to.is_set:
            # Set unqiue name based on object name and time, so that many nodes
            # can use the same dump directory
            dump_dir = self.dump_to.value / f"{self.name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
            dump_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Dumping data to {dump_dir}")

            for file in self.work_dir.iterdir():
                # Move directories, and copy files
                if file.is_dir():
                    shutil.move(file, dump_dir)
                else:
                    shutil.copy(file, dump_dir)

    def _get_executable(self) -> str:
        """Get the full path to the executable."""
        if self.required_callables:
            return shutil.which(self.required_callables[0])
        return None
