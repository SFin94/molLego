"""Module containing Molecule definiton."""
import sys
import numpy as np
import molLego.utilities.geom as geom

__ATOM_ELEMENTS__ = ['h',  'he', 'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne', 'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar', 'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu']


class Molecule():
    """
    Represents a Molecule from a calculation.

    Attributes
    ----------
    atom_ids : :class:`list of str`
        The atomic symbols of the atoms in the molecule.

    atom_number : :class:`int`
        The number of atoms in the molecule.
    
    charge : :class:`int`
        The formal charge of the molecule.
    
    geometry : :class:`numpy ndarray`
        A ``(N, 3)`` array of x, y, z coordinates for each atom.
        Where N is the number of atoms in the molecule.
    
    e : :class:`float`
        The energy of the molecule.

    parameters : :class:`dict`
        Where Key is the parameter identifier,
        and Value is the geometric parameter.
        E.g. O1-C2-O3: 180.00

    parser : `OutputParser`
        Parser to use for calculation output.
        
    """

    def __init__(self, output_file, parser):
        """
        Initialise a Molecule from calculation output file.

        Parameters
        ----------
        output_file : `str`
            The path to the calculation output file.

        parser : `OutputParser`
            Parser to use for calculation output.

        """
        # Set parser object.
        self.parser = parser(output_file)

        # Get base properties from calculation parser.
        properties = self.parser.get_properties()

        # Set attributes.
        self.atom_ids = self.parser.atom_ids
        self.atom_number = len(self.atom_ids)
        self.charge = properties['charge']
        self.geometry = properties['geom']
        self.e = properties['energy']
        self.parameters = {}

    def set_parameters(self, params):
        """
        Calculate geometric parameters (bonds, angles, dihedrals).

        Parameters saved in `dict`.
        Where Key is `str` of each atom_id and atom_index.
        E.g. C1-H2
        Value is the value of the geometric parameter.

        Parameters
        ----------
        params : :class:`iterable` of :class:`int`
            The atom indexes defining the geometric parameter.
            Can be either a nested iterable is multiple parameters,
            or a single iterable if only one parameter is required.

        """
        # Initialise values.
        param_keys = []
        param_vals = []

        # Handle if single parameter.
        if not any(isinstance(x, (list, tuple)) for x in params):
            params = [params]

        for par in params:
            # Set parameter key as atom_ID+atom_index'-'atom_ID+atom_index [-...-...]
            param_keys.append('-'.join([self.atom_ids[i] + str(i) for i in par]))
            param_vals = geom.calc_param(par, self.geometry)

        # Update parameter dict.
        self.parameters.update(dict(zip(param_keys, param_vals)))

    def get_atom_names(self, atom_index=None):
        """
        Yield unique name (atom_id+atom_index) of atom in molecule.

        Example would be `C1` for the C at index 1 in molecule.

        Parameters
        ----------
        atom_index : :class: `iterable` of :class:`int`
            The index(es) of the atoms to return names for.
            [default: ``None``] If ``None`` then returns for all
            atom in the molecule.
            Can be single `int` to call name for single atom.

        Yields
        -------
        :class: `int`
            Unique name of an atom made of atom_id+atom_index.

        """
        # Set to all atoms is atom_index is None.
        if atom_index is None:
            atom_index = range(self.atom_number)
        elif isinstance(atom_index, int):
            atom_index = (atom_index, )

        atomic_names = [self.atom_ids[i] + str(i) for i in atom_index]

        for atom_name in atomic_names:
            yield atom_name
        
    def get_atomic_numbers(self, atom_index=None):
        """
        Yield atomic number of atom.
        
        Parameters
        ----------
        atom_index : :class: `iterable` of :class:`int`
            The index(es) of the atoms to return names for.
            [default: ``None``] If ``None`` then returns for all
            atom in the molecule.
            Can be single `int` to call name for single atom.

        Yields
        -------
        :class: `int`
            The atomic number of an atom.

        """
        # Set to all atoms is atom_index is None.
        if atom_index is None:
            atom_index = range(self.atom_number)
        elif isinstance(atom_index, int):
            atom_index = (atom_index, )

        atomic_numbers = [int(__ATOM_ELEMENTS__.index(
                          self.atom_ids[i].lower())) + 1
                          for i in atom_index]

        for atom_number in atomic_numbers:
            yield atom_number

    def get_df_repr(self):
        """
        Create dict representation of Molecule for a DataFrame.

        Returns
        -------
        df_rep : `dict`
            Molecule properties in the format:
            {
                file_name   : path to parent output file,
                e : energy (units: calculation dependant),
                (optional)
                parameter key : parameter value
                [for all parameters in self.parameters]
            }

        """
        df_rep = {'file_name': self.parser.file_name,
                  'e': self.e}
        df_rep.update(self.parameters)

        return df_rep

    def reindex_molecule(self, reindex):
        """
        Reorder molecule geometry and atom list from a given mapping.

        Example
        -------
        Mapping is a list of the new positions for each atom.
        Indexes start at 0.

        For a molecule with 4 atoms with start order: [H C H O],
        
        To switch the position of the O and C atoms.
        Input reindex map: [0 3 2 1]
        Will keep H0 at 0; move C to position 3; 
        keep H2 at 2 and move O3 to 1.
        
        Returned order: [H O H C]

        Parameters
        ----------
        reindex : `list of int`
            List of new index positions for each atom.
    
        """
        self.geom = self.geom[reindex, :]
        self.atom_ids = [self.atom_ids[i] for i in reindex]

    def set_adjacency(self, distance=2.0):

        """
        Set adjacency matrix for the bond topology of a molecule from the geometry (cartesian coordinates) - uses simple distance metric to work out where a bond may be
        Sets class attributes:
         adjacency: :class:`numpy array` - dim: num. of atoms x num. of atoms; entries are 1 for an edge (bond)
        Update would use the vdws of the atoms to work out the bonding distances.
        """

        # Initialise variables
        self.adjacency = np.zeros((len(self.geom), len(self.geom)))

        # Calculates distance between atoms and if smaller than the distance tolerence a bond is assumed (matrix entry set to 1)
        for i, atom_i in enumerate(self.geom):
            for j, atom_j in enumerate(self.geom[i+1:]):
                self.adjacency[i, j+i+1] =  (geom.calc_dist(atom_i, atom_j) < distance)
        self.adjacency += self.adjacency.transpose()



