# Copyright (C) 2026  Shichao Wu, Alex Nitz
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Backend-independent pieces of time-delay interferometry algebra."""

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Term:
    """One delayed directed-link contribution to a TDI combination."""

    link: tuple
    coefficient: float
    operators: tuple


class TDICombination(Protocol):
    """Structural interface needed by dense and future sparse evaluators."""

    name: str

    def terms(self) -> Sequence[Term]:
        """Return the combination as delayed single-link terms."""
        ...

    def net_shifts(self, sample):
        """Return positive-delay shifts for every operator chain."""
        ...

    def data_delay_extent(self, sample):
        """Return the recorded-data history required in seconds."""
        ...

    def waveform_support(self, sample):
        """Return strain history required including one link response."""
        ...


def orthogonal_channels(x, y, z):
    """Return PyTDI/LDC-normalized ``(A, E, T)`` from ``(X, Y, Z)``."""
    return (
        (z - x) / np.sqrt(2),
        (x - 2 * y + z) / np.sqrt(6),
        (x + y + z) / np.sqrt(3),
    )
