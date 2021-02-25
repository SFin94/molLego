"""Module containing dummy parser to combine Molecules in to one type."""

import numpy as np
from molLego.parsers.parser import OutputParser
from molLego.utilities.utils import readlines_reverse, parse_mol_formula

class CombineError(Exception):
    """Raised when error in reading log file."""

class CombineParser(OutputParser):
    """
    Dummy parser to combine Molecules.

    Attributes
    ----------
    atoms : :class:`list of str`
        The atomic symbols of the atoms in the molecule.

    atom_number : :class:`int`
        The number of atoms in the molecule.
 
    charge : :class:`int`
        The charge of the molecule.

    file_name : :class:`list` of :class:`str`
        The paths to the parent log files.

    molecules : :class:`list` of :Molecule:
        The Molecules that have been combined.

    """
    
    def __init__(self, molecules):
        """
        Initialise parser properties from list of Molecules.

        Parameters
        ----------
        molecules : :class:`list` of :Molecule:
            The Molecules to be combined.

        """
        # Initialise variables.
        self.charge = 0
        self.file_name, self.atom_ids = [], []
        self.molecules = molecules
        
        # Combine file names and atom_ids. Set charge.
        for mol in molecules:
            self.file_name.append(mol.parser.file_name)
            [self.atom_ids.append(i) for i in mol.atom_ids]
            self.charge += mol.charge
        self.atom_number = len(self.atom_ids)

    def get_properties(self):
        """
        Get properties by combining those of exisiting Molecules.
        
        Returns
        -------
        :class:`dict`
            A dictionary containing target properties for the molecule.

        """
        # Check thermochemical properties present.

        thermo_quantities = ['e_therm', 'h', 'g', 's', 'zpe']
        present = np.asarray([hasattr(self.molecules[0], x) 
                              for x in thermo_quantities])
        thermo = {thermo_quantities[x]: 0.0 
                      for x in np.where(present)[0]}
        if hasattr(self.molecules[0], 't'):
            thermo['t'] = self.molecules[0].t

        # Combine properties across molecules.
        properties = {}
        i = 0
        energy = 0.0
        geometry = np.empty((self.atom_number, 3))
        for mol in self.molecules:

            # Sum thermochemical values.
            for x in thermo.keys():
                thermo[x] += getattr(mol, x)
            energy += mol.e

            # Combine geometries.
            geometry[i:mol.atom_number+i,:] = mol.geometry[:]
            i += mol.atom_number
    
        properties = {'geom': geometry, 'charge': self.charge, 
                      'energy': energy, 'thermo': thermo}

        return properties