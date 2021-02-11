"""Module containing Molecule definiton."""

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
        A ``(N, 3)`` array of x, y, z coorindates for each atom.
        Where N is the number of atoms in the molecule.
    
    e: :class:`float`
        The energy of the molecule.
        
    """

    def __init__(self, output_file, parser):
        """
        Initialise a Molecule from Gaussian Frequency calculation.

        Parameters
        ----------
        parser: `OutputParser`
            Parser to use for calculation output.
            [default: GaussianLog]
        
        output_file: `str`
            The path to the calculation output file.

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

    
    # Add representation strings?


    def set_parameters(self, param):
        """
        Calculate geometric parameters and set as property.
        WHETHER TO HAVE AS MULTIPLE OR NOT?
        
        Parameters
        ----------
        params : :class:`nested list of `int``
            List of the atom indexes of the geometric parameters.

        """
        # Set parameter key as atom_ID+atom_index'-'atom_ID+atom_index [-...-...]
        param_key = '-'.join([atom_ids[i] + str(i) for i in param])
        param_val = geom.calc_param(param, self.geom)
        # Up to - to change
        if 'parameters' in self.__dict__:
            self.parameters.update(dict(zip(param_keys, param_values)))
        else:
            self.parameters = dict(zip(param_keys, param_values))


    def set_adjacency(self, distance=2.0):

        """
        Set adjacency matrix for the bond topology of a molecule from the geometry (cartesian coordinates) - uses simple distance metric to work out where a bond may be
        Sets class attributes:
         adjacency: :class:`numpy array` - dim: num. of atoms x num. of atoms; entries are 1 for an edge (bond)
        """

        # Initialise variables
        self.adjacency = np.zeros((len(self.geom), len(self.geom)))

        # Calculates distance between atoms and if smaller than the distance tolerence a bond is assumed (matrix entry set to 1)
        for i, atom_i in enumerate(self.geom):
            for j, atom_j in enumerate(self.geom[i+1:]):
                self.adjacency[i, j+i+1] =  (geom.calc_dist(atom_i, atom_j) < distance)
        self.adjacency += self.adjacency.transpose()


    def set_atom_indexes(self):

        """Class method to convert list of atom ids to list of corresponding atom indexes

        Sets class attributes:
         atom_indexes: :class:`list` - atom ids as str entry

        """
        
        self.atom_indexes = [int(__ATOM_ELEMENTS__.index(i.lower()))+1 for i in self.atom_ids]


    def reindex_molecule(self, reindex):

        """Class method to reorder a molecules geometry and atom list based on a given mapping
        Updates class attributes:
         atoms: :class:`list of str` - atom IDs of the atoms in the molecule
         geom: :class:`ndarray` (dim: :class:atom_number, 3; float) - Results array of x, y, z coordinates for each atom in the molecule

        Parameters:
        reindex: list of int - list of new index positions
        """

        self.geom = self.geom[reindex, :]
        self.atom_ids = [self.atom_ids[i] for i in reindex]




