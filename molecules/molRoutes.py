import sys
import numpy as np
import pandas as pd
import utilities.analyseGaussLog as glog
import molecules


'''A module of routines that interface with the Molecule classes'''


def constructMols(systemFile, type='molecule'):

    '''Function which creates Molecule or MoleculeThermo object for each molecule in a system conf file
    
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

            if type == 'reaction':
                molecule.append(mol.initMol(molFile[-1], line.split()[2:], type='reaction'))
            else:
                molecules.append(mol.initMol(molFile[-1], type))

    return molNames, molFile, molecules


def moleculesToDataFrame(molecules, molNames=None, save=None):

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
        propDict = {'File': mol.logFile, 'E SCF (h)': mol.escf, 'Optimised': mol.optimised}

        if hasattr(mol, 'e'):
            propDict.update({'E': mol.e, 'H': mol.h, 'G': mol.g, 'S': mol.s, 'ZPE': mol.zpe})
            quantity = ['E', 'H', 'G']
        else:
            quantity = ['E SCF']
            propDict.update({'E SCF': mol.escf*2625.5})
        data.append(propDict)

        if hasattr(mol, 'parameters'):
            propDict.update(mol.parameters)

    if molNames == None:
        molNames = []
        [molNames.append(mol.logFile.split('/')[-1][:-4]) for mol in molecules]
        moleculeData = pd.DataFrame(data)
    moleculeData = pd.DataFrame(data, index=molNames)

    # Calculate the relative thermodynamic quantities
    moleculeData = calcRelative(moleculeData, quantities=quantity)

    # Writes dataframe to file if filename provided
    if save != None:
        moleculeData.to_csv(save + '.csv')
    return moleculeData


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
            molecule = molecules.initMolFromLog(logFile, optStep=step)
            scanMolecules.append(molecule)
            scanFiles.append(logFile)

        # Add parameter as an attribute for each scan molecule
        for scanMol in scanMolecules:
            scanMol.setParameters(parameters)
            

    return scanFiles, scanMolecules


def calcRelative(moleculeDataFull, molsToPlot=None, quantities=['E SCF (h)']):

    '''Function to process a dataframe of molecules to plot and calculates relative E (kJ/mol) [NB: Commented lines can also calcuate and normalised relative E]

        Parameters:
         moleculeDataFull: pandas dataFrame - full dataframe for molecules
         molsToPlot: List of str [Optional, default=None] - names (.log file) of conformers to plot from the dataFile
         energyCol: str [default='E SCF (h)'] - The quantitity to plot (str should match dataframe heading)

        Returns:
         confDiheds: pandas DataFrame - dataframe of the molecules to plot with realtive and normalised E columns for plotting
    '''

    # Subset amount of data frame to plot
    if molsToPlot != None:
        moleculeData = moleculeDataFull.reindex(molsToPlot)
    else:
        moleculeData = moleculeDataFull

    # Calculate relative and normalised quantities
    for q in quantities:
        moleculeData['Relative '+q] = moleculeData[q] - moleculeData[q].min()

    return moleculeData


def sumMolecules(*args):

    '''Function that adds two molecules together to creat a new one, e.g. for a reactant or product set

    Parameters:
     args: Molecule objects - the molecules to be added

    Returns:
     newMolecule - ::class:: object for a molecule

    '''

    # Set sums for quantities and empty lists
    escfSum = 0.0
    atomList, logFiles = [], []
    if hasattr(args[0], 'e'):
        type = 'thermo'
        eSum, gSum, hSum, sSum, zpeSum = 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        type = 'molecule'

    # Add values for each molecule to quantity sums
    for mol in args:

        logFiles.append(mol.logFile)
        atomList.append(mol.atoms)
        escfSum += mol.escf

        if type == 'thermo':
            try:
                eSum += mol.e
                hSum += mol.h
                gSum += mol.g
                sSum += mol.s
                zpeSum += mol.zpe
            except AttributeError:
                print('Molecule does not have correct thermodynamic values to be summed')

    # Instantiate molecule class with summed values
    if type == 'thermo':
        newMolecule = molecules.MoleculeThermo(logFiles, escfSum, molGeom=None, atomIDs=atomList, thermo=[eSum, hSum, gSum, sSum, zpeSum])
    else:
        newMolecule = molecules.Molecule(logFiles, escfSum, molGeom=None, atomIDs=atomList)

    return newMolecule
