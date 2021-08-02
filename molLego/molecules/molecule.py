"""Module containing Molecule definiton."""
import sys
import numpy as np
import scipy
import networkx as nx
# from rdkit.Chem import GetPeriodicTable
import molLego.utilities.geom as geom

__ATOM_ELEMENTS__ = ['h',  'he', 'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne', 'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar', 'k',  'ca', 'sc', 'ti', 'v', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu']

__COVALENT_RADII__ = {'h': 0.23, 'he': 0.93, 'li': 0.68, 'be': 0.35, 'b': 0.83, 'c': 0.68, 'n': 0.68, 'o': 0.68, 'f': 0.64, 'ne': 1.12, 'na': 0.97, 'mg': 1.1, 'al': 1.35, 'si': 1.2, 'p': 0.75, 's': 1.02, 'cl': 0.99, 'ar': 1.57, 'k': 1.33, 'ca': 0.99, 'sc': 1.44, 'ti': 1.47, 'v': 1.33, 'cr': 1.35, 'mn': 1.35, 'fe': 1.34, 'co': 1.33, 'ni': 1.5, 'cu': 1.52, 'zn': 1.45, 'ga': 1.22, 'ge': 1.17, 'as': 1.21, 'se': 1.22, 'br': 1.21, 'kr': 1.91, 'rb': 1.47, 'sr': 1.12, 'y': 1.78, 'zr': 1.56, 'nb': 1.48, 'mo': 1.47, 'tc': 1.35, 'ru': 1.4, 'rh': 1.45, 'pd': 1.5, 'ag': 1.59, 'cd': 1.69, 'in': 1.63, 'sn': 1.46, 'sb': 1.46, 'te': 1.47, 'i': 1.4, 'xe': 1.98, 'cs': 1.67, 'ba': 1.34, 'la': 1.87, 'ce': 1.83, 'pr': 1.82, 'nd': 1.81, 'pm': 1.8, 'sm': 1.8, 'eu': 1.99, 'gd': 1.79, 'tb': 1.76, 'dy': 1.75, 'ho': 1.74, 'er': 1.73, 'tm': 1.72, 'yb': 1.94, 'lu': 1.72, 'hf': 1.57, 'ta': 1.43, 'w': 1.37, 're': 1.35, 'os': 1.37, 'ir': 1.32, 'pt': 1.5, 'au': 1.5, 'hg': 1.7, 'tl': 1.55, 'pb': 1.54, 'bi': 1.54, 'po': 1.68, 'at': 1.7, 'rn': 2.4, 'fr': 2.0, 'ra': 1.9, 'ac': 1.88, 'th': 1.79, 'pa': 1.61, 'u': 1.58, 'np': 1.55, 'pu': 1.53}
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

        Parameters are saved in `dict`.
        Where Key is `str` of each atom_id and atom_index.
        E.g. C1-H2
        Value is the value of the geometric parameter.

        Parameters
        ----------
        params : :class:`iterable` of :class:`int`
            The atom indexes defining the geometric parameter.
            Can be either a nested iterable if multiple parameters,
            or a single iterable if only one parameter required.

        """
        # Initialise values.
        param_keys = []
        param_vals = []

        # Handle if single parameter.
        if not any(isinstance(x, (list, tuple)) for x in params):
            params = [params]

        # Set parameter key as atom_ID+atom_index'-'atom_ID+atom_index [-...-...]
        for par in params:
            param_keys.append('-'.join([self.atom_ids[i] + str(i) for i in par]))
        
        # Calculate parameters.
        param_vals = geom.calc_param(params, self.geometry)

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
        Mapping is a list where the current atom id is placed in the new position 
        that the atom will be moved to.
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
        self.geometry = self.geometry[reindex, :]
        self.atom_ids = [self.atom_ids[i] for i in reindex]
        
        # Reset adjacency for new index order.
        if hasattr(self, 'adjacency'):
            self.set_adjacency()

    def _calc_covr(self, atom_i, atom_j):
        return (__COVALENT_RADII__[atom_i.lower()]
                + __COVALENT_RADII__[atom_j.lower()])

    def _find_paths(self, start_node, mol_graph):
        """Find paths in graph from start node by depth first search."""
        path = [start_node]
        molecule_paths = []
        for edge in nx.dfs_edges(mol_graph, source=start_node):
            if edge[0] == start_node:
                molecule_paths.append(path)
                path = [edge[1]]
            else:
                path.append(edge[1])
        molecule_paths.append(path)
        return molecule_paths

    def _stack_paths(self, molecule_paths):
        """Set index list of molecule paths from longest to shortest."""
        new_index = []
        # Stack paths in order of length for new index list.
        while molecule_paths:
            path_lengths = [len(path) for path in molecule_paths]
            next_path = path_lengths.index(min(path_lengths))
            new_index.extend(molecule_paths.pop(next_path))
        return new_index

    def get_bonds(self):
        """
        Compute bond list in molecule from adjacency matrix.

        Returns
        -------
        :numpy:`array`
            List of bonds in terms of atom indexes.
        
        """
        if not hasattr(self, 'adjacency'):
            self.set_adjacency()

        # Find all non zero entries and return as tuples.
        return np.transpose(np.nonzero(self.adjacency))

    def index_by_paths(self, start_node=0):
        """
        Reindex a molecule using order of paths from a starting node.

        Parameters
        ----------
        start_node: `int`
            The index (0 index) of the starting atom to use in the 
            reference molecule. 

        """
        # Get bond list for graph edges.
        bonds = self.get_bonds()

        # Initialise graph.
        mol_graph = nx.Graph()
        mol_graph.add_edges_from(bonds)

        # Initialise index list.
        new_index = []
        while len(new_index) < len(self.atom_ids):
            # Set new start atom if disconnected graph or new mol.
            if len(new_index) > 0:
                # Find atoms not in index and set new start node.
                start_set = set(range(len(self.atom_ids))).difference(set(new_index))
                start_node = list(start_set)[0]

            # Find paths in molecule and add to new index list.
            molecule_paths = self._find_paths(start_node, mol_graph)
            new_index += self._stack_paths(molecule_paths)

        # Reindex molecule.        
        self.reindex_molecule(new_index)

    def set_distance(self):
        """Set distance matrix."""
        self.distance = scipy.spatial.distance_matrix(self.geometry, 
                                                      self.geometry)

    def set_adjacency(self, dist_factor=1.4):
        """
        Set adjacency matrix where 1 shows covalent bonds.

        Parameters
        ----------
        dist_factor : :class:`float`
            Scale factor for sum of covalent radii.
        """
        # Calculate distance matrix.
        self.set_distance()

        # Create threshold matrix.
        dist_tol = np.zeros((self.atom_number, self.atom_number))
        for i in range(self.atom_number):
            for j in range(self.atom_number):
                dist_tol[i, j] = self._calc_covr(self.atom_ids[i], 
                                                 self.atom_ids[j])*dist_factor
        
        # Set adjacency matrix.
        self.adjacency = (self.distance < dist_tol).astype(int)
        np.fill_diagonal(self.adjacency, 0)
        
    def to_xyz(self, file_name=None):
        """
        Write xyz file of molecule.
        
        Parameters
        ----------
        file_name : :class:`str`
            Output file name/path. Optional.
            [Default: ``None``] If ``None`` then uses molecule input
            file name with .xyz extension.
        """
        if file_name is None:
            file_name = self.parser.file_name.split('/')[-1].split('.')[0] + '.xyz'
        if file_name.split('.')[-1] != 'xyz':
            file_name += '.xyz'
        
        # Write xyz file.
        with open(file_name, "w+") as out_file:
            print(self.atom_number, file=out_file)
            print(f"Structure of {file_name} from {self.parser.file_name}", file=out_file)
            for i, atom in enumerate(self.geometry):
                print(f"{self.atom_ids[i]:<4}{atom[0]:>10f}{atom[1]:>10f}{atom[2]:>10f}",
                    file=out_file)
                


