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

    def __init__(self, logfile, mol_energy, mol_geom, atom_ids, optimised=False):

        self.file_name = logfile
        self.escf = mol_energy
        self.atom_ids = atom_ids
        self.atom_number = len(atom_ids)
        self.geom = mol_geom
        self.optimised = optimised

    def set_parameters(self, parameters, gauss_index=False):

        '''Class method to set dict of parameters as additional attribute
        Sets class attributes:
         parameters: :class:`dict` - key is the parameter key and the value is the calculated parameter value from the molecules geometry
        
        Parameters:
         parameters: dict - parameter key (atoms): parameter atom indexes; by default should be pythonic index
         gauss_index: bool - flag that can be set to True if parameters are given as gaussian indexes (start at 1) and not pythonic indexes (start at 0) 
        '''

        param_keys = list(parameters.keys())

        # If gaussian indexes then transform to python index
        if gauss_index == True:
            for param in param_keys:
                parameters[param] = [i-1 for i in parameters[param]]

        param_values = geom.calc_param(list(parameters.values()), self.geom)
        if 'parameters' in self.__dict__:
            self.parameters.update(dict(zip(param_keys, param_values)))
        else:
            self.parameters = dict(zip(param_keys, param_values))


    def reindex_molecule(self, reindex):

        '''Class method to reorder a molecules geometry and atom list based on a given mapping
        Updates class attributes:
         atoms: :class:`list of str` - atom IDs of the atoms in the molecule
         geom: :class:`ndarray` (dim: :class:atom_number, 3; float) - Results array of x, y, z coordinates for each atom in the molecule
    
        Parameters:
        reindex: list of int - list of new index positions
        '''

        self.geom = self.geom[reindex, :]
        self.atoms = [self.atoms[i] for i in reindex]


class MoleculeThermo(Molecule):

    '''Class attributes:
     [Inherited from parent class Molecule: atom_number, atom_ids, geom, optimised, escf]

        e: :class:`float` - thermally corrected total energy of the molecule (kJ/mol)
        h: :class:`float` - thermally corrected total enthalpy of the molecule (kJ/mol)
        g: :class:`float` - thermally corrected total Gibbs free energy of the molecule (kJ/mol)
        s: :class:`numpy.ndarray` - total entropy of the molecule (kJ/mol)
        zpe: :class:`float` - zero-point energy of the molecule (kJ/mol)
    '''

    def __init__(self, logfile, mol_energy, mol_geom, atom_ids, optimised, thermo):

        # Set thermodynamic values (energy, enthalpy, Gibbs free energy, entropy, zpe) for molecule
        super().__init__(logfile, mol_energy, mol_geom, atom_ids, optimised)
        self.e, self.h, self.g, self.s, self.zpe = thermo


class ReactionPath():

    '''Class attributes:

        reac_steps: :class:`ThermoMolecule object` - steps of the reaction profile
        reac_step_names: :class:`list` - str identifiers for each reaction step in the profile
        reac_coord: :class:`` - floats between 0 and 1 of the reaction coordinate for each step
         NB: This can either be calculated assuming equal spacing or passed explicitly
     '''

    def __init__(self, molecules, step_names, reac_coord=None):

        self.reac_steps = molecules
        self.reac_step_names = step_names

        # Calculate the reaction coordinates for the path (assuming linear if not inputted)
        if reac_coord == None:
            self.reac_coord = np.linspace(0, 1, len(step_names))
        else:
            scale = reac_coord[-1] - reac_coord[0]
            self.reac_coord = (reac_coord - reac_coord[0])/scale


def init_mol_from_log(logfile, opt_steps=[1], parameters=None):

    '''Function that initiates a molecule or moleculeThermo object from a gaussian log file

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

    # Initiate Molecule or MoleculeThermo object for each molecule
    for i, mol in enumerate(mol_results.values()):
        if 'thermo' in list(job.job_property_flags.keys()):
            molecule = MoleculeThermo(job.file_name, mol['energy'], mol['geom'], job.atom_ids, mol['opt'], mol['thermo'])
        else:
            molecule = Molecule(job.file_name, mol['energy'], mol['geom'], job.atom_ids, mol['opt'])

        # Set parameters for each molecule if given
        if parameters != None:
            molecule.set_parameters(parameters)
   
        # Test if single or multiple molecules to return or append
        if len(mol_results) == 1:
            return molecule
        else:
            molecules.append(molecule)
    return molecules


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
