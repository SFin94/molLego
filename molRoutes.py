import sys
import numpy as np
import pandas as pd

import utilities.analyseGaussLog as glog
import molecules as mol


def constructMols(systemFile, type='molecule'):

    '''Function which creates thermochemistry objects for each molecule
    Parameters:
     systemFile: str - name of the system file containing the molecule names/keys and the log files to be parsed
            The file should be formatted:
                molKey mol_output.log
        Lines can be commented out with a #

    Returns:
     molNames: list of str - molecule names/keys for each molecule in file [molKey in system file]
     molFile: list of str - molecules log file for each molecule in file
     molecules: list of Molecule/MoleculeThermo objects for each molecule in system file
    '''
    # Read in system file
    with open(systemFile) as file:
        input = file.read().splitlines()

    # Creates a molecule or thermo object for each of the molecules inputted
    molecules, molNames, molFile = [], [], []
    for line in input:
        if line[0] != '#':
            molNames.append(line.split()[0])
            molFile.append(line.split()[1])
            molecules.append(mol.initMol(molFile[-1], type))

    return molNames, molFile, molecules


