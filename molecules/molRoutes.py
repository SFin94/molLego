import sys
import numpy as np
import pandas as pd
import molLego.utilities.analyseGaussLog as glog
import molLego.utilities.geom as geom
import molLego.molecules.molecules as molecules


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
     mols: list of Molecule/MoleculeThermo objects for each molecule in system file
    '''

    # Read in system file
    with open(systemFile) as file:
        input = file.read().splitlines()

    # Creates a molecule or thermo object for each of the molecules inputted
    mols, molNames, molFiles = [], [], []
    for line in input:
        if line[0] != '#':
            molNames.append(line.split()[0])
            molFiles.append(line.split()[1].split(','))

            # Create moleucle object for first input file and sum all moleucles together if multiple files
            mols.append(molecules.initMolFromLog(molFiles[-1][0], type))
            for mFile in molFiles[-1][1:]:
                extraMol = molecules.initMolFromLog(mFile, type)
                mols[-1] = sumMolecules(mols[-1], extraMol)

    return molNames, mols


def moleculesToDataFrame(mols, molNames=None, save=None, min=None):

    '''Function which creates a dataframe for all of the molecules and can write to a csv

    Parameters:
     mols: list of Molecule or MoleculeThermo objects - instances for each molecule
     molNames [optional, default=None]: list - molecule names/keys
     save [optional, default=None]: str - name of file to write dataframe to (without csv extension)
    '''

    # Create a dataframe of molecule attributes depending on object type (Molecule or MoleculeThermo)
    data = []
    for ind, mol in enumerate(mols):
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
        [molNames.append(mol.logFile.split('/')[-1][:-4]) for mol in mols]
        moleculeData = pd.DataFrame(data)
    moleculeData = pd.DataFrame(data, index=molNames)

    # Calculate the relative thermodynamic quantities
    moleculeData = calcRelative(moleculeData, quantities=quantity, min=min)

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

    Returns:
     scanMolecules: List of Molecule objects for each step of scan
    '''

    scanMolecules = []

    # Parse in tracked parameters if set
    if trackedParams != None:
        parameters = parseTrackedParams(trackedParams)
    else:
        parameters = {}

    for logFile in args:
        # Get scanInfo - modRed input
        scanInfo = glog.pullScanInfo(logFile)
        parameters[scanInfo['paramKey']] = scanInfo['atomInd']
        # Test to see if param is the same here - else flag warning

        for step in range(1, scanInfo['nSteps']+2):
            molecule = molecules.initMolFromLog(logFile, optStep=step)
            molecule.setParameters(parameters)
            scanMolecules.append(molecule)

    return scanMolecules, scanInfo


def initRigidScan(*args, trackedParams=None):

    '''Function that generates a list of molecule objects from a rigid scan file

    Parameters:
     args: str - gaussian log files of scan results
     trackedParams: [optional] str - file with tracked parameters in (gaussian indexes)

    Returns:
     scanMolecules: List of Molecule objects for each step of scan
    '''

    scanMolecules = []

    # Parse in tracked parameters if set
    if trackedParams != None:
        parameters = parseTrackedParams(trackedParams)
    else:
        parameters = {}

    for logFile in args:
        scanVariables, scanSteps, initialzMat = glog.pullRigidScanInfo(logFile)

    # Pull scan parameter info from initial z matrix (python index)
    for atomInd, atomzMat in enumerate(initialzMat):
        line = atomzMat.split()
        for sV in scanVariables:
            if sV in line:
                indList = [atomInd]
                for  i in range(1, line.index(sV), 2):
                    indList.append(int(line[i]) - 1)
                parameters[sV] = indList

    # Calculate tracked and scan parameters, and energy for each step - scan to mol to df
    for logFile in args:
       for step in range(1, scanSteps+1):
            molecule = molecules.initMolFromLog(logFile, optStep=step, type='spe')
            molecule.setParameters(parameters)
            scanMolecules.append(molecule)

    return scanMolecules, scanVariables


def calcRelative(moleculeDataFull, molsToPlot=None, quantities=['E SCF'], min=None):

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
        if min != None:
            zero = moleculeData[q][min]
        else:
            zero = moleculeData[q].min()
        moleculeData['Relative '+q] = moleculeData[q] - zero

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
    optimised = True

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

        if mol.optimised == False:
            optimised = False
            print('Warning, one molecule in complex is not optimised')

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
        newMolecule = molecules.MoleculeThermo(logFiles, escfSum, molGeom=None, atomIDs=atomList, optimised=optimised, thermo=[eSum, hSum, gSum, sSum, zpeSum])
    else:
        newMolecule = molecules.Molecule(logFiles, escfSum, molGeom=None, atomIDs=atomList, optimised=optimised)

    return newMolecule


def initReactionProfile(reacStepNames, reacSteps, paths):

    '''Function that creates a reaction profile object for a reaction path

    Parameters:
     reacStepNames: list - str identifiers of the unique steps on the reaction profile
     reacSteps: list - ThermoMolecular objects of the unique steps on the reaction profile
     paths: list - indexes of the steps making up each reaction path in the profile

    Returns:
     reactionProfile: list of :class:objects -  List of ReactionPath objects containing the molecules in the path
    '''

    # Set initial variables
    reactionProfile = []

    for reactionPath in paths:
        reactantsNode = reactionPath[0]
        pathMolecules = [reacSteps[reactantsNode]]
        pathNames = [reacStepNames[reactantsNode]]

        # For each seperate path create a ReactionPath object
        for pathStep in reactionPath[1:]:
            if pathStep == reactantsNode:
                reactionProfile.append(molecules.ReactionPath(pathMolecules, pathNames))
                pathMolecules = []
                pathNames = []
            pathMolecules.append(reacSteps[pathStep])
            pathNames.append(reacStepNames[pathStep])
        reactionProfile.append(molecules.ReactionPath(pathMolecules, pathNames))

    return reactionProfile


def constructReactionPath(systemFile, molNames=None):

    # Read in system file
    with open(systemFile) as file:
        input = file.read().splitlines()

    # Parse molNames from system file if not already created
    if molNames == None:
        molNames = []
        for line in input:
            if line[0] != '#':
                molNames.append(line.split()[0])

    # Set neighbour list form system file
    # Might not need branches, numSteps or even stepNeighbours
    numSteps = len(input)
    branches = 1
    stepNeighbours = []
    for line in input:
        if line[0] != '#':
            if len(line.split()) > 2:
                stepNeighbours.append(line.split()[2].split(','))
                branches += len(stepNeighbours[-1]) - 1
            else:
                stepNeighbours.append([])

    # Set adjacency matrix
    adjacency = np.zeros((numSteps, numSteps))
    for node, edgeSet in enumerate(stepNeighbours):
        for edge in edgeSet:
            adjacency[node, molNames.index(edge)] = 1

    # Calculate path list from adjacency
    pathList = []
    reactantNodes = np.nonzero(np.sum(adjacency, axis=0) == 0)[0]
    for rNode in reactantNodes:
        pathList.append(trackReactionPath(rNode, adjacency))

    return pathList, stepNeighbours


def trackReactionPath(currentStep, adjacency, path=[]):

    path = path + [currentStep]
    if np.count_nonzero(adjacency[currentStep,:]) == 0:
        return path

    paths = []
    nextPaths = np.nonzero(adjacency[currentStep,:])[0]
    for nP in nextPaths:
        nextStep = trackReactionPath(nP, adjacency, path)
        for nS in nextStep:
            paths.append(nS)
    return paths


def reacProfileToDataFrame(reactionProfile, save=None, min=None):

    rProfileData = pd.DataFrame()

    # For each reaction path create dataframe then append additional columns
    for rPathInd, reactionPath in enumerate(reactionProfile):
        rPathData = moleculesToDataFrame(reactionPath.reacSteps, reactionPath.reacStepNames, min=min)
        rPathData['Reaction coordinate'] = reactionPath.reacCoord
        rPathData['Reaction path'] = [rPathInd]*len(reactionPath.reacStepNames)

        rProfileData = rProfileData.append(rPathData)


    # Writes dataframe to file if filename provided
    if save != None:
        rProfileData.to_csv(save + '.csv')

    return rProfileData

