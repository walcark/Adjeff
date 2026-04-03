"""Define a custom logger for Adjeff and the associated configuration class.

The logger can :
- Filter redondant messages,

"""

from typing import Any


class MultilineConsoleRenderer:
    """Structlog renderer that prints each context field on its own line.

    Produces output like::

        2026-04-02 08:52:22 [debug    ] Aggregated parameters.
            das     = {'wl': ['wl'], 'aot': ['x', 'y']}
            scalars = []
            vectors = ['wl', 'aot', 'rh']

    Parameters
    ----------
    indent : int
        Number of spaces to indent field lines.
    pad_keys : bool
        If True, align the ``=`` signs by padding key names to the same width.
    colors : bool
        If True, colorize the level token using ANSI codes.
    """

    _LEVEL_COLORS = {
        "debug": "\033[36m",  # cyan
        "info": "\033[32m",  # green
        "warning": "\033[33m",  # yellow
        "error": "\033[31m",  # red
        "critical": "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"

    def __init__(
        self,
        indent: int = 4,
        pad_keys: bool = True,
        colors: bool = True,
    ) -> None:
        self._indent = " " * indent
        self._pad_keys = pad_keys
        self._colors = colors

    def __call__(
        self,
        logger: Any,
        method: str,
        event_dict: dict[str, Any],
    ) -> str:
        """Process the log and applies a multi-line rendering."""
        timestamp = event_dict.pop("timestamp", "")
        level = event_dict.pop("level", method)
        event = event_dict.pop("event", "")

        level_str = f"{level:<8}"
        if self._colors:
            color = self._LEVEL_COLORS.get(level.lower(), "")
            level_str = f"{color}{level_str}{self._RESET}"

        header = f"{timestamp} [{level_str}] {event}"

        if not event_dict:
            return header

        if self._pad_keys:
            width = max(len(k) for k in event_dict)
            lines = [
                f"{self._indent}{k:<{width}} = {v}"
                for k, v in event_dict.items()
            ]
        else:
            lines = [f"{self._indent}{k} = {v}" for k, v in event_dict.items()]

        return header + "\n" + "\n".join(lines)
