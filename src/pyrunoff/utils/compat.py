"""Compatibility helpers for optional runtime dependencies."""

import functools
import warnings

try:
    from numba import jit

    HAS_NUMBA = True

except ImportError:
    HAS_NUMBA = False

    def jit(*args, **kwargs):
        """Return a warning-emitting no-op replacement for Numba's ``jit``.

        :param args: Positional decorator options, accepted for compatibility.
        :param kwargs: Keyword decorator options, accepted for compatibility.
        :return: A decorator that leaves the function uncompiled.
        """

        def decorator(func):
            """Wrap a function with the Numba-unavailable warning."""

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                """Warn that the function is running without compilation."""
                warnings.warn(
                    f"Numba not found. '{func.__name__}' is running in "
                    "pure Python mode. "
                    "Install with 'pip install numba' or "
                    "'pyrunoff[with_numba]' for a 10-100x speedup.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return func(*args, **kwargs)

            return wrapper

        return decorator
