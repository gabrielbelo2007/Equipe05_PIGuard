"""
Reexporta todas as funcionalidades de filters.py.
"""

try:  # execução como módulo (python -m ...)
    from . import filters as _filters
except ImportError:  # execução direta / import via sys.path
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    import filters as _filters

__all__ = list(
    getattr(
        _filters,
        "__all__",
        [name for name in dir(_filters) if not name.startswith("_")],
    )
)
globals().update({name: getattr(_filters, name) for name in __all__})
