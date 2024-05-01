"""Utilities for creating BioSimSpace Nodes"""

import sys
from pathlib import Path
from typing import Callable

from maize.core.interface import Input, Output
from maize.core.node import Node
from maize.core.workflow import Workflow, expose
from maize.steps.io import Return
from maize.utilities.chem import Isomer

from .enums import BSSEngine


class _ClassProperty:
    """
    Create a class-level property.

    Example:
    --------
    >>> class MyClass:
    >>>     _class_attr = 0

    >>>     @ClassProperty
    >>>     def class_attr(cls):
    >>>         return cls._class_attr
    """

    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


def create_engine_specific_nodes(
    abstract_base_node: Node,
    module: str,
    engines: list[BSSEngine] = [Engine for Engine in BSSEngine],
    create_exposed_workflows: bool = False,
) -> None:
    """
    Create engine-specific nodes for BioSimSpace from an abstract base node.
    The abstract base node should be named "_<Protocol>Base", e.g. "_MinimiseBase".

    Parameters
    ----------
    abstract_base_node : Node
        The abstract base node from which to create the engine-specific nodes.
    module : str
        The module to create the nodes in.
    engines : list[BSSEngine], optional
        The engines to create nodes for, by default all engines defined in the BSSEngine Enum.
    create_exposed_workflows : bool, optional
        Whether to create engine-specific exposed workflows for the created nodes, by default False.
    """
    module_dict = sys.modules[module].__dict__

    for engine in engines:
        protocol_name = abstract_base_node.__name__.split("Base")[0][1:]
        doctring_desc = protocol_name if protocol_name != "Production" else "Run production MD on"
        class_name = f"{protocol_name}{engine.class_name}"
        docstring = f"""
        {doctring_desc} the system using {engine.name.capitalize()} through BioSimSpace.

        Notes
        -----
        Install with `mamba create -f env.yaml`.

        References
        ----------
        L. O. Hedges et al., JOSS, 2019, 4, 1831.
        L. O. Hedges et al., LiveCoMS, 2023, 5, 2375–2375.
        """
        module_dict[class_name] = type(
            class_name,
            (abstract_base_node,),
            {"bss_engine": engine, "__doc__": docstring},
        )

        if create_exposed_workflows:
            fn_name = f"{protocol_name.lower()}_{engine.function_name}_exposed"
            module_dict[fn_name] = expose(get_workflow_fn(module_dict[class_name]))


def get_workflow_fn(node: Node) -> Callable[[], Workflow]:
    """
    Get a function returning a simple workflow
    for a single node from a single node (which must be
    a subclass of _BioSimSpaceBase).

    Parameters
    ----------
    node : Node
        The node to create the workflow from.

    Returns
    -------
    Callable[[], Workflow]
        A function returning a simple workflow for the node.
    """

    def workflow_fn() -> Workflow:
        flow = Workflow(name=node.__name__)
        bss_node = flow.add(node)
        # We have to connect the output to something
        retu = flow.add(Return[tuple[Path, Path]], name="Return")
        flow.connect(bss_node.out, retu.inp)
        flow.map(bss_node.inp)
        filter_parameters = [
            "batch_options",
            "commands",
            "modules",
            "python",
            "scripts",
            "save_name",
        ]
        flow.map(
            *[
                param
                for param in bss_node.parameters.values()
                if param.name not in filter_parameters
            ]
        )

        # Make sure that the save name is always set when run through the CLI
        # TODO: Figure out why this fails to set the default for the CLI
        flow.combine_parameters(bss_node.save_name, default=f"{node.__name__.lower()}_output")

        return flow

    # The below steps are required to get the workflow summary to
    # show when -h is supplied to the CLI
    # Rename workflow_fn according to the node name
    workflow_fn.__name__ = f"{node.__name__.lower()}_workflow"
    # Add the docstring from the node to the workflow function
    workflow_fn.__doc__ = node.__doc__

    return workflow_fn


def get_ligand_from_system(
    system: "BioSimSpace.Sandpit.Exscientia._SireWrappers.System", ligand_name: str = "LIG"
) -> "BioSimSpace.Sandpit.Exscientia._SireWrappers.Molecule":
    """
    Find the ligand in the system and return it. Should only be called within
    the `run` method of a node.

    Parameters
    ----------
    system : BioSimSpace.Sandpit.Exscientia._SireWrappers.System
        The system containing the ligand.
    ligand_name : str
        The name of the ligand to find. Default is 'LIG'.

    Returns
    -------
    BioSimSpace.Sandpit.Exscientia._SireWrappers.Molecule
        The ligand molecule.
    """
    import BioSimSpace.Sandpit.Exscientia as BSS

    try:
        lig = system.search(f"resname {ligand_name}").molecules()[0]
    except IndexError:
        raise ValueError(f"No ligand called '{ligand_name}' found in the input system.")

    return lig


def mark_ligand_for_decoupling(
    system: "BioSimSpace.Sandpit.Exscientia._SireWrappers.System", ligand_name: str = "LIG"
) -> "BioSimSpace.Sandpit.Exscientia._SireWrappers.System":
    """
    Find the ligand in the system, makr it for decoupling,
    and return the updated ligand. Should only be called within
    the `run` method of a node.

    Parameters
    ----------
    system : BioSimSpace.Sandpit.Exscientia._SireWrappers.System
        The system containing the ligand.
    ligand_name : str
        The name of the ligand to mark for decoupling. Default is 'LIG'.

    Returns
    -------
    BioSimSpace.Sandpit.Exscientia._SireWrappers.System
        The updated system with the ligand marked for decoupling.
    """
    import BioSimSpace.Sandpit.Exscientia as BSS

    # Get the ligand
    lig = get_ligand_from_system(system, ligand_name)

    # Decouple the ligand
    lig_decoupled = BSS.Align.decouple(lig)
    system.updateMolecule(system.getIndex(lig), lig_decoupled)

    return system


def get_ligand_smiles(
    system: "BioSimSpace.Sandpit.Exscientia._SireWrappers.System",
    ligand_name: str = "LIG",
) -> str:
    """
    Get the SMILES string for the ligand in the system. Should only be called within
    the `run` method of a node.

    Parameters
    ----------
    system : BioSimSpace.Sandpit.Exscientia._SireWrappers.System
        The system containing the ligand.
    ligand_name : str
        The name of the ligand to get the SMILES for. Default is 'LIG'.

    Returns
    -------
    str
        The SMILES string for the ligand.
    """
    import BioSimSpace.Sandpit.Exscientia as BSS

    # Get the ligand
    lig = get_ligand_from_system(system, ligand_name)

    # Get the SMILES
    return lig._sire_object.smiles()


def rename_lig(
    bss_system: "BioSimSpace._SireWrappers._system.System", new_name: str = "LIG"
) -> None:
    """
    Rename the ligand in a BSS system. Should only be called within
    the `run` method of a node.

    Parameters
    ----------
    bss_system : BioSimSpace._SireWrappers._system.System
        The BSS system.
    new_name : str
        The new name for the ligand.
    Returns
    -------
    None
    """
    import BioSimSpace as _BSS
    from BioSimSpace._SireWrappers import Molecule as _Molecule
    from sire.legacy import Mol as _SireMol

    # Ensure that we only have one molecule
    if len(bss_system) != 1:
        raise ValueError("BSS system must only contain one molecule when renaming.")

    # Extract the sire object for the single molecule
    mol = _Molecule(bss_system[0])
    mol_sire = mol._sire_object

    # Create an editable version of the sire object
    mol_edit = mol_sire.edit()

    # Rename the molecule and the residue to the supplied name
    resname = _SireMol.ResName(new_name)  # type: ignore
    mol_edit = mol_edit.residue(_SireMol.ResIdx(0)).rename(resname).molecule()  # type: ignore
    mol_edit = mol_edit.edit().rename(new_name).molecule()

    # Commit the changes and update the system
    mol._sire_object = mol_edit.commit()
    bss_system.updateMolecule(0, mol)


class IsomerToSDF(Node):
    """
    Convert a Isomer object to a path to an SDF.
    """

    # Input
    inp: Input[Isomer] = Input()
    """
    Input isomer object.
    """

    out: Output[Path] = Output()
    """
    Path to the output SDF file.
    """

    def run(self) -> None:

        # Get the sdf input
        isomer = self.inp.receive()

        # Save to an sdf file
        sdf_path = Path("isomer.sdf")
        isomer.to_sdf(sdf_path)

        # Send the path to the output
        self.out.send(sdf_path)
