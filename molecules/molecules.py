import numpy as np
import molLego.utilities.analyseGaussLog as glog
import molLego.utilities.geom as geom

'''
Script containing class definitions for base class Molecule and sub class Thermomolecule
'''

class Molecule():

    '''
    Class attributes
        logFile: :class:`str` - filepath/name of parent log file
        escf: :class:`float` - SCF energy of molecule (a.u.)
        numAtoms: :class:`int` - The number of atoms in the molecule
        atoms: :class:`list of str` - atom IDs of the atoms in the molecule
        atomCoords: :class:`ndarray` (dim: :class:numAtoms, 3; float) - Results array of x, y, z coordinates for each atom in the molecule

    '''

    def __init__(self, logFile, molEnergy, molGeom, atomIDs, optimised):

        self.logFile = logFile
        self.escf = molEnergy
        self.atoms = atomIDs
        self.numAtoms = len(atomIDs)
        self.atomCoords = molGeom
        self.optimised = optimised

    def setParameters(self, parameters, gaussIndex=False):

        '''Class method to set dict of parameters as additional attribute
        
        Parameters:
         parameters: dict - parameter key (atoms): parameter atom indexes; by default should be pythonic index
         gaussIndex: bool - flag that can be set to True if parameters are given as gaussian indexes (start at 1) and not pythonic indexes (start at 0) 
        '''

        paramKeys = list(parameters.keys())

        # If gaussian indexes then transform to python index
        if gaussIndex == True:
            for param in paramKeys:
                parameters[param] = [pVal -1 for pVal in parameters[param]]

        paramValues = geom.calcParam(list(parameters.values()), self.atomCoords)
        if 'parameters' in self.__dict__:
            self.parameters.update(dict(zip(paramKeys, paramValues)))
        else:
            self.parameters = dict(zip(paramKeys, paramValues))


    def reindexMolecule(self, reindex):

        '''Class method to reorder a molecules geometry and atom list based on a given mapping
    
        Parameters:
        reindex: list of int - list of new index positions
        '''

        self.atomCoords = self.atomCoords[reindex, :]
        self.atoms = [self.atoms[i] for i in reindex]


class MoleculeThermo(Molecule):

    '''Class attributes:
     [Inherited from parent class Molecule: numAtoms, atomIDs, atomCoords, optimised, escf]

        eTherm: :class:`float` - thermally corrected total energy of the molecule (kJ/mol)
        h: :class:`float` - thermally corrected total enthalpy of the molecule (kJ/mol)
        g: :class:`float` - thermally corrected total Gibbs free energy of the molecule (kJ/mol)
        s: :class:`numpy.ndarray` - total entropy of the molecule (kJ/mol)
        zpe: :class:`float` - zero-point energy of the molecule (kJ/mol)
    '''

    def __init__(self, logFile, molEnergy, molGeom, atomIDs, optimised, thermo):


        # Set thermodynamic values (energy, enthalpy, Gibbs free energy, entropy, zpe) for molecule
        super().__init__(logFile, molEnergy, molGeom, atomIDs, optimised)
        self.e, self.h, self.g, self.s, self.zpe = thermo


class ReactionPath():

    '''Class attributes:

        reacSteps: :class:`ThermoMolecule object` - steps of the reaction profile
        reacStepNames: :class:`list` - str identifiers for each reaction step in the profile
        reacCoords: :class:`` - floats between 0 and 1 of the reaction coordinate for each step
         NB: This can either be calculated assuming equal spacing or passed explicitly
     '''

    def __init__(self, molecules, stepNames, reacCoord=None):

        self.reacSteps = molecules
        self.reacStepNames = stepNames

        # Calculate the reaction coordinates for the path (assuming linear if not inputted)
        if reacCoord == None:
            self.reacCoord = np.linspace(0, 1, len(stepNames))
        else:
            scale = reacCoord[-1] - reacCoord[0]
            self.reacCoord = (reacCoord - reacCoord[0])/scale


def initMolFromLog(logFile, type='molecule', optStep=1):

    '''Function that initiates a molecule or moleculeThermo object from a gaussian log file assuming that the final geometry is wanted (e.g. from an opt or freq calculation)

    Parameters:
     logFile: str - name of the gaussian log file
     type: str - whether a molecule or moleculeThermo object is to be created

    Returns:
     molecule: :class:object for a molecule
    '''

    # Parse all properties from gaussian log file - currently don't set optstep or mp2
    if type.lower() == 'spe':
        molEnergy = glog.pullEnergy(logFile, optStep, SPE=True)[0]
        molGeom, optimised = glog.pullGeom(logFile, optStep, SPE=True)
    else:
        molEnergy = glog.pullEnergy(logFile, optStep)[0]
        molGeom, optimised = glog.pullGeom(logFile, optStep)
    atomIDs = glog.pullAtomIDs(logFile)

    # If thermochemistry wanted, parse additional information
    if type.lower()[:5] == 'therm':
        thermo = glog.pullThermo(logFile)
        molecule = MoleculeThermo(logFile, molEnergy, molGeom, atomIDs, optimised, thermo)
    else:
        molecule = Molecule(logFile, molEnergy, molGeom, atomIDs, optimised)

    return molecule


def initMolFromDF(dfFileEntry, type='molecule', geom=False, optStep=1):

    '''Function that initiates a molecule or moleculeThermo object from a prexisting data file
    Parameters:
     datafile: str - name of the gaussian log file
     type: str - whether a molecule or moleculeThermo object is to be created

    Returns:
     molecule: :class:object for a molecule
    '''

    rawDataFrame = pd.read_csv(dfFile)
    logFile = rawDataFrame['File']
    molEnergy = rawDataFrame['E SCF (h)']

    # Parse all properties from gaussian log file - currently don't set optstep or mp2
    if geom == True:
        molGeom, optimised = glog.pullGeom(logFile, optStep)
        atomIDs = glog.pullAtomIDs(logFile)
    else:
        molGeom = None
        atomIDs = None

    '''order of thermo'''
    if type != 'molecule':
        thermo = rawDataFrame['E'], rawDataFrame['H'], rawDataFrame['G'], rawDataFrame['S'], rawDataFrame['ZPE']
        molecule = MoleculeThermo(logFile, molEnergy, molGeom, atomIDs, thermo)
    else:
        molecule = Molecule(logFile, molEnergy, molGeom, atomIDs)

    return molecule
