"""Utilities for creating BioSimSpace Nodes"""


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
