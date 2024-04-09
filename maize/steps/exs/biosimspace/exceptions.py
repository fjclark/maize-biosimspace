"""Custom exceptions for BioSimSpace Nodes."""

__all__ = ["BioSimSpaceNullSystemError"]


class BioSimSpaceNullSystemError(Exception):
    """Raised when a BioSimSpace system is None."""

    def __init__(self, message="System is None"):
        super().__init__(message)
