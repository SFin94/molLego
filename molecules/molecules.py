import sys
import numpy as np

import molLego.utilities.analyseGaussLog as glog
import molLego.utilities.geom as geom

'''
Script containing class definitions for base class Molecule and sub class Thermomolecule
'''

class Molecule():

    '''
    Class attributes
        file_name: :class:`str` - filepath/name of parent calculation file
        escf: :class:`float` - SCF energy of molecule (a.u.)
        atom_number: :class:`int` - The number of atoms in the molecule
        atoms: :class:`list of str` - atom IDs of the atoms in the molecule
        geom: :class:`ndarray` (dim: :class:atom_number, 3; float) - Results array of x, y, z coordinates for each atom in the molecule

    '''

    def __init__(self, input_file, mol_energy, mol_geom, atom_ids, optimised=False):

        self.file_name = input_file
        self.escf = mol_energy
        self.atom_ids = atom_ids
        self.atom_number = len(atom_ids)
        self.geom = mol_geom
        self.optimised = optimised

    def set_parameters(self, parameters, gauss_index=False):
        """
        Class method to set dict of parameters as additional attribute
        Sets class attributes:
         parameters: :class:`dict` - key is the parameter key and the value is the calculated parameter value from the molecules geometry
        
        Parameters:
         parameters: dict - parameter key (atoms): parameter atom indexes; by default should be pythonic index
         gauss_index: bool - flag that can be set to True if parameters are given as gaussian indexes (start at 1) and not pythonic indexes (start at 0) 
        """
        param_keys = list(parameters.keys())

        # If gaussian indexes then transform to python index
        if gauss_index == True:
            for param in param_keys:
                parameters[param] = [i-1 for i in parameters[param]]

        # Calculates parameter value from molecule geometry and updates parameter dict
        param_values = geom.calc_param(list(parameters.values()), self.geom)
        if 'parameters' in self.__dict__:
            self.parameters.update(dict(zip(param_keys, param_values)))
        else:
            self.parameters = dict(zip(param_keys, param_values))


    def set_adjacency(self, distance=2.0):

        '''Sets adjacency matrix for the bond topology of a molecule from the geometry (cartesian coordinates) - uses simple distance metric to work out where a bond may be
        Sets class attributes:
         adjacency: :class:`numpy array` - dim: num. of atoms x num. of atoms; entries are 1 for an edge (bond)
        '''

        # Initialise variables
        self.adjacency = np.zeros((len(self.geom), len(self.geom)))

        # Calculates distance between atoms and if smaller than the distance tolerence a bond is assumed (matrix entry set to 1)
        for i, atom_i in enumerate(self.geom):
            for j, atom_j in enumerate(self.geom[i+1:]):
                self.adjacency[i, j+i+1] =  (geom.calc_dist(atom_i, atom_j) < distance)
        self.adjacency += self.adjacency.transpose()


    def set_atom_indexes(self):

        '''Class method to convert list of atom ids to list of corresponding atom indexes

        Sets class attributes:
         atom_indexes: :class:`list` - atom ids as str entry

        '''

        # List of atoms - index matches atomic mass - 1
        atom_id_index = ['h',  'he', 'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne', 'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar', 'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu']

        self.atom_indexes = [int(atom_id_index.index(i.lower()))+1 for i in self.atom_ids]


    def reindex_molecule(self, reindex):

        '''Class method to reorder a molecules geometry and atom list based on a given mapping
        Updates class attributes:
         atoms: :class:`list of str` - atom IDs of the atoms in the molecule
         geom: :class:`ndarray` (dim: :class:atom_number, 3; float) - Results array of x, y, z coordinates for each atom in the molecule
    
        Parameters:
        reindex: list of int - list of new index positions
        '''

        self.geom = self.geom[reindex, :]
        self.atom_ids = [self.atom_ids[i] for i in reindex]


class MoleculeThermo(Molecule):

    '''Class attributes:
     [Inherited from parent class Molecule: atom_number, atom_ids, geom, optimised, escf]

        e: :class:`float` - thermally corrected total energy of the molecule (kJ/mol)
        h: :class:`float` - thermally corrected total enthalpy of the molecule (kJ/mol)
        g: :class:`float` - thermally corrected total Gibbs free energy of the molecule (kJ/mol)
        s: :class:`float` - total entropy of the molecule (kJ/mol)
        zpe: :class:`float` - zero-point energy of the molecule (kJ/mol)

    Additional Parameters:
     thermo: list of floats - themochemistry values in the order: ZPE, thermally corrected E, H and G, and TS 
    '''

    def __init__(self, input_file, mol_energy, mol_geom, atom_ids, optimised, zpe=0.0, e=0.0, h=0.0, g=0.0, s=0.0):

        # Set thermodynamic values (energy, enthalpy, Gibbs free energy, entropy, zpe) for molecule
        super().__init__(input_file, mol_energy, mol_geom, atom_ids, optimised)

        # Set thermodynamic quantities
        self.zpe = zpe
        self.e = e
        self.h = h
        self.g = g
        self.s = s


def init_mol_from_log(logfile, opt_steps=None, parameters=None):

    '''Function that initiates a Molecule or MoleculeThermo object from a gaussian log file

    Parameters:
     logfile: str - name of the gaussian log file
     optsteps: list of int - target optimised/geometries to create molecule object for from gaussian log file (relaxed/rigid scan)
     parameters: dict - parameter key (atoms): parameter atom indexes; by default should be pythonic index

    Returns:
     molecule: :class:object for a molecule
    '''

    # Initialise class for log file - sets properties of the calculations to parse
    job = glog.GaussianLog(logfile)
    molecules = []

    # Parse all properties for the calculation type
    mol_results = job.pull_properties(opt_steps=opt_steps)

    # Initiate Molecule or MoleculeThermo object for each molecule.
    for i, mol in enumerate(mol_results.values()):
        
        # Set opt property to False if not existing attribute.
        if not hasattr(mol, 'opt'):
            mol['opt'] = False
        
        try:
            # Process dict of thermochemistry results to pass ZPE, thermally corrected E, H, G and TS to init
            molecule = MoleculeThermo(job.file_name, mol['energy'], mol['geom'], job.atom_ids, mol['opt'], zpe=mol['thermo']['ZPE'], e=mol['thermo']['E'], h=mol['thermo']['H'], g=mol['thermo']['G'], s=mol['thermo']['S'])
        except:
            molecule = Molecule(job.file_name, mol_energy=mol['energy'], mol_geom=mol['geom'], atom_ids=job.atom_ids, optimised=mol['opt'])

        # Set parameters for each molecule if given
        if parameters != None:
            molecule.set_parameters(parameters)
   
        # Test if single or multiple molecules to return or append
        if len(mol_results) == 1:
            return molecule
        else:
            molecules.append(molecule)
    return molecules


def init_mol_from_xyz(xyzfile, parameters=None):

    '''Function that initiates a Molecule object from an xyz file

    Parameters:
     xyzfile: str - name of the gaussian log file
     parameters: dict - parameter key (atoms): parameter atom indexes; by default should be pythonic index

    Returns:
     molecule: list of :class:objects for a molecule 
    '''

    # Intitialise variables
    molecule = []

    # Open and iterate through log file
    with open(xyzfile, 'r') as input:
        atom_number = int(input.readline())
        for line in input:

            # If multiple molecules then process next one with new atom number
            if len(molecule) >= 1:
                atom_number = int(line.split()[0])
                line = input.__next__()

            # Process energy if comment is number or set as title
            try:
                energy = float(line.split()[0])
            except:
                energy = 0.0

            # Intialise variables
            atom_ids = []
            atom_coords = []
            
            # Pull geometry and atom ids
            for i in range(atom_number):
                line = input.__next__()
                atom_ids.append(line.split()[0])
                xyz = np.asarray([float(i) for i in line.split()[1:]])
                atom_coords.append(xyz)
            geometry = np.asarray(atom_coords)

            # Initialise molecule
            molecule.append(Molecule(xyzfile, mol_energy=energy, mol_geom=geometry, atom_ids=atom_ids))

    return molecule


# def initMolFromDF(data_file, geom=False, optStep=1):

#     '''Function that initiates a molecule or moleculeThermo object from a prexisting data file
#     Parameters:
#      datafile: str - name of the gaussian log file
#      type: str - whether a molecule or moleculeThermo object is to be created

#     Returns:
#      molecule: :class:object for a molecule
#     '''

#     raw_data_frame = pd.read_csv(df_file)
#     log_file = raw_data_frame['File']
#     mol_energy = raw_data_frame['E SCF (h)']

#     # Parse all properties from gaussian log file - currently don't set optstep or mp2
#     if geom == True:
#         molGeom, optimised = glog.pullMolecule(logfile, target='geom', optStep=optStep)
#         atomIDs = glog.pullAtomIDs(logfile)
#     else:
#         molGeom = None
#         atomIDs = None

#     '''order of thermo'''
#     if type != 'molecule':
#         thermo = rawDataFrame['E'], rawDataFrame['H'], rawDataFrame['G'], rawDataFrame['S'], rawDataFrame['ZPE']
#         molecule = MoleculeThermo(logfile, molEnergy, molGeom, atomIDs, thermo)
#     else:
#         molecule = Molecule(logfile, molEnergy, molGeom, atomIDs)

#     return molecule
