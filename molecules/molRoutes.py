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


def moleculesToDataFrame(molFiles, molecules, molNames=None, save=None):

    '''Function which creates a dataframe for all of the molecules and can write to a csv
    Parameters:
     molNames: list - molecule names/keys
     molFiles: list - corresponding gaussian log files the data originates from
     molecules: list of MoleculeThermo objects - MoleculeThermo instances for each molecule
     fileName [optional]: str - name of file to write dataframe to (without csv extension)
    '''
    # Create a dataframe of molecule attributes depending on object type (Molecule or MoleculeThermo)
    data = []
    for ind, mol in enumerate(molecules):
        propDict = {'File': molFiles[ind], 'E SCF (h)': mol.escf}

        if hasattr(mol, 'eTherm'):
            propDict.update({'E': mol.eTherm, 'H': mol.h, 'G': mol.g, 'S': mol.s, 'zpe': mol.zpe})
            quantity = ['E', 'H', 'G']
        else:
            quantity = ['E SCF']
            propDict.update({'E SCF': mol.escf*2625.5})
        data.append(propDict)

        if hasattr(mol, 'parameters'):
            propDict.update(mol.parameters)

    if molNames != None:
        molDF = pd.DataFrame(data, index=molNames)
    else:
        molDF = pd.DataFrame(data)

    # Calculate the relative thermodynamic quantities
    for q in quantity:
        molDF['Relative ' + q] = molDF[q] - molDF.min()[q]

    # Writes dataframe to file if filename provided
    if save != None:
        molDF.to_csv(save + '.csv')
    return molDF


def parseTrackedParams(systemFile):

    '''Function which parses any additional parameters to be tracked from an input file

        Input:
         inputfile: str - name of input .txt file which contains any additional parameters to be tracked across the scan

         Format of input file:
             paramName (atomTypes) atomInd1 atomInd2 [atomInd3 atomInd4]
             E.g. OPSC 3 1 2 7

        Returns:
         trackedParams: dict:
                         key: str - paramName
                         value: list of ints - [atomIndexes]
    '''

    # Initialise empty dict for params
    trackedParams = {}
    # Parse in file and seperate the indexes from the parameter ID and save as an entry to the dict
    with open(systemFile, 'r') as input:
        for el in input:
            param = el.strip().split(' ')
            indexes = [int(ind)-1 for ind in param[1:]]
            trackedParams[param[0]] = indexes
    return trackedParams


def initScan(*args, trackedParams=None):

    '''Function that generates a list of molecule objects from a scan file

    Parameters:
     args: str - gaussian log files of scan results
     trackedParams: [optional] str - file with tracked parameters in (gaussian indexes)

    '''

    scanMolecules = []
    scanFiles = []

    if trackedParams != None:
        parameters = parseTrackedParams(trackedParams)
    else:
        parameters = {}

    for logFile in args:
        # Get scanInfo - modRed input
        scanInfo = glog.pullScanInfo(logFile)
        parameters[scanInfo['paramKey']] = scanInfo['atomInd']
        # Test to see if param is the same here - else flag warning

        for step in range(1, scanInfo['nSteps']):
            molecule = mol.initMol(logFile, optStep=step)
            scanMolecules.append(molecule)
            scanFiles.append(logFile)

        # Add parameter as an attribute for each scan molecule
        for scanMol in scanMolecules:
            scanMol.setParameters(parameters)

    return scanFiles, scanMolecules
