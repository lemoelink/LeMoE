"""
session_store — stub (versión light)
La telemetría a disco ha sido eliminada en esta rama.
"""


def push_context(*args, **kwargs) -> None:
    """No-op: telemetría desactivada en light."""
    pass


def flush_session() -> None:
    """No-op: telemetría desactivada en light."""
    pass
