"""Compatibility helpers for optional runtime dependencies."""

import functools
import warnings

NUMBA_MISSING_MESSAGE = (
    "Numba not found; numerical kernels are running in pure Python mode. "
    "Install numba for a 10-100x speedup."
)

# Whether the missing-Numba warning has been issued in this process.
_warned = False


def _warn_numba_missing():
    """Issue the missing-Numba warning the first time a kernel runs."""
    global _warned
    if _warned:
        return
    _warned = True
    warnings.warn(NUMBA_MISSING_MESSAGE, RuntimeWarning, stacklevel=3)


def python_jit(*args, **kwargs):
    """Return a no-op replacement for Numba's ``jit``.

    The decorated function runs uncompiled; the first call to any such
    function warns once per process that Numba is not installed.

    :param args: Positional decorator options, accepted for compatibility.
    :param kwargs: Keyword decorator options, accepted for compatibility.
    :return: A decorator that leaves the function uncompiled.
    """

    def decorator(func):
        """Wrap a function with the one-time Numba-unavailable warning."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Warn on first use, then run the function uncompiled."""
            _warn_numba_missing()
            return func(*args, **kwargs)

        return wrapper

    return decorator


try:
    from numba import jit

    HAS_NUMBA = True

except ImportError:
    HAS_NUMBA = False
    jit = python_jit
