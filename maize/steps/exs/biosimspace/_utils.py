"""Utilities for creating BioSimSpace Nodes"""

import sys

from maize.core.node import Node

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
    module_name: str,
    exclude_engines: list[BSSEngine] = [BSSEngine.TLEAP, BSSEngine.NONE],
) -> None:
    """
    Create engine-specific nodes for BioSimSpace from an abstract base node.
    The abstract base node should be named "_<Protocol>Base", e.g. "_MinimiseBase".

    Parameters
    ----------
    abstract_base_node : Node
        The abstract base node from which to create the engine-specific nodes.
    module_name : str
        The name of the module where the new classes should be added.
    exclude_engines : list[BSSEngine], optional
        A list of engines to exclude from the creation of the nodes, by default [BSSEngine.TLEAP, BSSEngine.NONE]
    """
    module_dict = sys.modules[module_name].__dict__

    for engine in BSSEngine:
        if engine in exclude_engines:
            continue
        protocol_name = abstract_base_node.__name__.split("Base")[0][1:]
        doctring_desc = protocol_name if protocol_name != "Production" else "Run production MD on"
        class_name = f"{protocol_name}{engine.class_name}"
        print(f"Creating {class_name} node.")
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
