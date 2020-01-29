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

    def setParameters(self, parameters):

        '''Class function to set dict of parameters as additional attribute'''
        paramKeys = list(parameters.keys())
        paramValues = geom.calcParam(list(parameters.values()), self.atomCoords)
        self.parameters = dict(zip(paramKeys, paramValues))


class MoleculeThermo(Molecule):

    '''Class attributes:
     [Inherited from parent class Molecule: numAtoms, atomIDs, atomCoords, optimised, escf]

        eTherm: :class:`float` - thermally corrected total energy of the molecule (kJ/mol)
        h: :class:`float` - thermally corrected total enthalpy of the molecule (kJ/mol)
        g: :class:`float` - thermally corrected total Gibbs free energy of the molecule (kJ/mol)
        s: :class:`float` - total entropy of the molecule (kJ/mol)
        zpe: :class:`float` - zero-point energy of the molecule (kJ/mol)
    '''

    def __init__(self, logFile, molEnergy, molGeom, atomIDs, thermo):


        # Set thermodynamic values (energy, enthalpy, Gibbs free energy, entropy, zpe) for molecule
        super().__init__(logFile, molEnergy, molGeom, atomIDs, optimised)
        self.e, self.h, self.g, self.s, self.zpe = thermo

#
#    def setReactionCoord(self,):
#
#
#    def setNeighbours(self, neighbourList):



def initMolFromLog(logFile, type='molecule', optStep=1):

    '''Function that initiates a molecule or moleculeThermo object from a gaussian log file assuming that the final geometry is wanted (e.g. from an opt or freq calculation)

    Parameters:
     logFile: str - name of the gaussian log file
     type: str - whether a molecule or moleculeThermo object is to be created

    Returns:
     molecule: :class:object for a molecule
    '''

    # Parse all properties from gaussian log file - currently don't set optstep or mp2
    molEnergy = glog.pullEnergy(logFile, optStep)[0]
    molGeom, optimised = glog.pullGeom(logFile, optStep)
    atomIDs = glog.pullAtomIDs(logFile)

    # If thermochemistry wanted, parse additional information
    if type != 'molecule':
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
        print(thermo)
        molecule = MoleculeThermo(logFile, molEnergy, molGeom, atomIDs, thermo)
    else:
        molecule = Molecule(logFile, molEnergy, molGeom, atomIDs)

    return molecule
