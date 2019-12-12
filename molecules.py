import numpy as np
import utilities.analyseGaussLog as glog

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

    def __init__(self, logFile, molEnergy, molGeom, atomIDs):

        self.logFile = logFile
        self.escf = molEnergy
        self.atoms = atomIDs
        self.numAtoms = len(atomIDs)
        self.atomCoords = molGeom

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
        super().__init__(logFile, molEnergy, molGeom, atomIDs)
        self.eTherm, self.h, self.g, self.s, self.zpe = thermo


#class Scan(Molecule):
#
#    '''Class that is an object of Molecules - one for each scan step; can either be from scan file or several stand alone files?'''
#
#    super().__init__(logFile)


def initMol(logFile, type='molecule'):

    '''Function that initiates a molecule or moleculeThermo object assuming that the final geometry is wanted (e.g. from an opt or freq calculation)

    Parameters:
     logFile: str - name of the gaussian log file
     type: str - whether a molecule or moleculeThermo object is to be created

    Returns:
     molecule: :class:object for a molecule
    '''

    # Parse all properties from gaussian log file - currently don't set optstep or mp2
    numAtoms = glog.countAtoms(logFile)
    molEnergy = glog.pullEnergy(logFile)[0]
    molGeom, optimised = glog.pullGeom(logFile)
    atomIDs = glog.pullAtomIDs(logFile)

    # If thermochemistry wanted, parse additional information
    if type != 'molecule':
        thermo = glog.pullThermo(logFile)
        molecule = MoleculeThermo(logFile, molEnergy, molGeom, atomIDs, thermo)
    else:
        molecule = Molecule(logFile, molEnergy, molGeom, atomIDs)

    return molecule

