"""An Enum for the supported engines in BioSimSpace."""

from enum import auto

from maize.utilities.utilities import StrEnum

__all__ = ["BSSEngine", "LegType", "Ensemble", "StageType"]


class BSSEngine(StrEnum):
    """Available BioSimSpace Engines."""

    GROMACS = auto()
    SANDER = auto()
    PMEMD = auto()
    PMEMD_CUDA = auto()
    OPENMM = auto()
    SOMD = auto()
    NAMD = auto()
    TLEAP = auto()
    NONE = auto()

    @property
    def class_name(self) -> str:
        """Get the class name for the engine."""
        return "".join([word.capitalize() for word in self.name.split("_")])

    @property
    def function_name(self) -> str:
        """Get the function name for the engine."""
        return self.name.lower()


class LegType(StrEnum):
    """Leg types e.g. free ligand or complex."""

    FREE = auto()
    BOUND = auto()

    @property
    def class_name(self) -> str:
        """Get the class name for the leg."""
        return self.name.capitalize()

    @property
    def leg_name(self) -> str:
        """Get the leg name for the leg."""
        return self.name.lower()


class StageType(StrEnum):
    """Stage types"""

    RESTRAIN = auto()
    DISCHARGE = auto()
    VANISH = auto()

    @property
    def class_name(self) -> str:
        """Get the class name for the stage."""
        return self.name.capitalize()

    @property
    def stage_name(self) -> str:
        """Get the stage name for the stage."""
        return self.name.lower()

    @property
    def perturbation_type(self) -> str:
        """Get the perturbation type for the stage."""
        pert_types = {
            "RESTRAIN": "restraint",
            "DISCHARGE": "discharge_soft",
            "VANISH": "vanish_soft",
        }

        return pert_types[self.name]


_ENGINE_CALLABLES = {
    BSSEngine.GROMACS: ["gmx"],
    BSSEngine.SANDER: ["sander"],
    BSSEngine.PMEMD: ["pmemd"],
    BSSEngine.PMEMD_CUDA: ["pmemd.cuda"],
    BSSEngine.OPENMM: ["sire_python"],
    BSSEngine.SOMD: ["somd"],
    BSSEngine.NAMD: ["namd"],
    BSSEngine.TLEAP: ["tleap"],
    BSSEngine.NONE: [],
}
"""A mapping between the BioSimSpace engine and the required callables for each engine."""


class Ensemble(StrEnum):
    """Available ensembles for BioSimSpace simulations."""

    NVT = auto()
    NPT = auto()