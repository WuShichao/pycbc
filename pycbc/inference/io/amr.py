# Copyright (C) 2026  Shichao Wu
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""File handling for the ``amr`` sampler."""

from .dynesty import DynestyFile


class AMRFile(DynestyFile):
    """Output of the ``amr`` sampler.

    The posterior samples are written as equal-weight raw samples with a
    ``logwt`` column, read back the way the nested samplers' are. The final
    boxes of the refinement are kept in ``sampler_info/cells``.
    """
    name = 'amr_file'

    def write_sampler_metadata(self, sampler):
        self.attrs['sampler'] = sampler.name
        if self.sampler_group not in self.keys():
            self.create_group(self.sampler_group)
        group = self[self.sampler_group]
        for key in ('initial_divisions', 'max_level', 'loglikelihood_step',
                    'mass_fraction', 'max_evaluations', 'posterior_samples',
                    'seed'):
            group.attrs[key] = getattr(sampler, key)
        sampler.model.write_metadata(self)

    def write_cells(self, cells):
        """Writes the final boxes to ``sampler_info/cells``."""
        path = self.sampler_group + '/cells'
        if path in self:
            del self[path]
        group = self.create_group(path)
        for key, value in cells.items():
            group[key] = value
