"""Module containing parent parser class."""

import numpy as np
from molLego.utilities.utils import readlines_reverse, parse_mol_formula

# class OutputFileError(Exception):
    # """Rasied when error in reading log file."""

class OutputParser():
    """
    Parser base class for calculation output file.

    Attributes
    ----------
    atom_ids : :class:`list of str`
        The IDs of the atoms in the molecule.

    atom_number : :class:`int`
        The number of atoms in the molecule.
 
    charge : :class:`int`
        The charge of the molecule.

    elements : :class:`list of str`
        The list of elements present in the molecule.

    file_name : :class:`str`
        The path to the parent log file.

    method : :class:`str`
        The method of the calculation (functional, etc.).

    normal_termination : :class:`Bool`
        ``True`` if normal termination. Otherwise, ``False``.

    """
    "INIT NEEDED?"
    def __init__(self, output_file):
        raise NotImplementedError

    def _check_normal_termination(self):
        """Check if calculation terminated correctly."""
        raise NotImplementedError

    def _pull_atom_ids(self):
        """Pull the atom IDs from calculation output file."""
        raise NotImplementedError

    def _pull_geometry(self):
        """Pull the geometry from calculation output file."""
        raise NotImplementedError

    def _pull_energy(self):
        """Pull the energy from calculation output file."""
        raise NotImplementedError

    def get_properties(self):
        """Get all properties from calculation output file."""
        raise NotImplementedError
