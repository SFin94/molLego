import sys
import numpy as np
import pandas as pd

import molLego.utilities.analyseGaussLog as glog
import molLego.utilities.geom as geom
import molLego.molecules.molecules as molecules


'''A module of routines that interface with the Molecule classes'''


def construct_mols(system_file):

    '''Function which creates Molecule or MoleculeThermo object for each molecule in a system conf file

    Parameters:
     system_file: str - name of the system file containing the molecule names/keys and the log files to be parsed
            The file should be formatted:
                mol_key mol_output.log
        Lines can be commented out with a #

    Returns:
     mol_names: list of str - molecule names/keys for each molecule in file [mol_key in system file]
     mols: list of :Molecule:/:MoleculeThermo: objects for each molecule in system file
    '''

    # Read in system conf file
    with open(system_file) as input_file:
        input = input_file.read().splitlines()

    # Processes contents of file, creating amolecule or thermo object for each molecule
    mols, mol_names, mol_files = [], [], []
    for line in input:
        if line[0] != '#':
            mol_names.append(line.split()[0])
            mol_files.append(line.split()[1].split(','))

            # Create moleucle object for first input file and sum moleucles together if multiple files
            mols.append(molecules.init_mol_from_log(mol_files[-1][0]))
            for m_file in mol_files[-1][1:]:
                extra_mol = molecules.init_mol_from_log(m_file)
                mols[-1] = sum_molecules(mols[-1], extra_mol)

    return mol_names, mols


def mols_to_dataframe(mols, mol_names=None, save=None, min=None):

    '''Function which creates a dataframe for all of the molecules and can write to a csv

    Parameters:
     mols: list of Molecule or MoleculeThermo objects - instances for each molecule
     mol_names [optional, default=None]: list - molecule names/keys
     save [optional, default=None]: str - name of file to write dataframe to (without csv extension)
     min [optional, default=None]: str - name of the molecule to calculate all values relative too. If not defined then relative values are calculated w.r.t. lowest value for each quantity.

    Returns:
     molecule_df: pandas dataframe - dataframe of all molecules with realtive quantities calcualted
    '''

    # Create a dataframe of molecule attributes depending on object type (Molecule or MoleculeThermo)
    data = []
    for ind, mol in enumerate(mols):
        properties = {'File': mol.file_name, 'E SCF (h)': mol.escf, 'Optimised': mol.optimised}

        if hasattr(mol, 'e'):
            properties.update({'E': mol.e, 'H': mol.h, 'G': mol.g, 'S': mol.s, 'ZPE': mol.zpe})
            quantity = ['E', 'H', 'G']
        else:
            quantity = ['E SCF']
            properties.update({'E SCF': mol.escf*2625.5})
        data.append(properties)

        if hasattr(mol, 'parameters'):
            properties.update(mol.parameters)
    if mol_names == None:
        mol_names = []
        [mol_names.append(mol.file_name.split('/')[-1][:-4]) for mol in mols]
        molecule_df = pd.DataFrame(data)
    molecule_df = pd.DataFrame(data, index=mol_names)

    # Calculate the relative thermodynamic quantities
    molecule_df = calcRelative(molecule_df, quantities=quantity, min=min)

    # Writes dataframe to file if filename provided
    if save != None:
        molecule_df.to_csv(save + '.csv')
    return molecule_df


def parse_tracked_params(system_file):

    '''Function which parses any additional parameters to be tracked from an input file

        Input:
         system_file: str - name of input .txt file which contains any additional parameters to be tracked across the scan

         Format of input file:
             param_name (atom_types) atom1_ind atom2_ind [atom3_ind atom4_ind]
             E.g. OPSC 3 1 2 7

        Returns:
         tracked_params: dict:
                         key: str - param_name
                         value: list of ints - [atom_indexes]
    '''

    # Initialise empty dict for params
    tracked_params = {}
    # Parse in file and seperate the indexes from the parameter ID and save as an entry to the dict
    with open(system_file, 'r') as input:
        for el in input:
            param = el.strip().split(' ')
            indexes = [int(ind)-1 for ind in param[1:]]
            tracked_params[param[0]] = indexes
    return tracked_params


def init_scan(*args, tracked_params=None):

    '''Function that generates a list of molecule objects from a scan file

    Parameters:
     args: str - gaussian log files of scan results
     tracked_params: [optional] str - file with tracked parameters in (gaussian indexes)

    Returns:
     scan_molecules: List of Molecule objects for each step of scan
    '''

    # Inititalise variables
    scan_molecules = []

    # Parse in tracked parameters if set
    if tracked_params != None:
        parameters = parse_tracked_params(tracked_params)
    else:
        parameters = {}

    for i, input_file in enumerate(args):

        # Get scanInfo - modRed input
        scan_file = glog.GaussianLog(input_file)
        scan_info = scan_file.set_scan_info()
        
        # Set scan parameter in parameters dict and range of opt steps in file
        parameters[scan_info['param_key']] = scan_info['atom_inds']
        opt_steps = list(range(1, scan_info['num_steps']+2))
        
        # Create molecule object for each scan step in input files
        if i == 0:
            scan_molecules = molecules.init_mol_from_log(input_file, opt_steps=opt_steps, parameters=parameters)
        else:
            scan_molecules.append(molecules.init_mol_from_log(input_file, opt_steps=opt_steps, parameters=parameters))

    return scan_molecules


def init_rigid_scan(*args, tracked_params=None):

    '''Function that generates a list of molecule objects from a rigid scan file

    Parameters:
     args: str - gaussian log files of scan results
     trackedParams: [optional] str - file with tracked parameters in (gaussian indexes)

    Returns:
     scanMolecules: List of Molecule objects for each step of scan
    '''

    scan_molecules = []

    # Parse in tracked parameters if set
    if tracked_params != None:
        parameters = parse_tracked_params(tracked_params)
    else:
        parameters = {}

    for i, input_file in enumerate(args):
    for logile in args:
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
            molecule = molecules.init_mol_from_log(logFile, optStep=step, type='spe')
            molecule.setParameters(parameters)
            scanMolecules.append(molecule)

    return scanMolecules, scanVariables


def calcRelative(moleculeDataFull, molsToPlot=None, quantities=None, min=None):

    '''Function to process a dataframe of molecules to plot and calculates relative E SCF (kJ/mol) or Relative E/G/H if thermodynamic properties given

        Parameters:
         moleculeDataFull: pandas dataFrame - full dataframe for molecules
         molsToPlot: List of str [optional, default=None] - names (.log file) of conformers to plot from the dataFile
         quantities: list of str [optional, default=None] - The quantitity/ies to plot (str should match dataframe heading). If None, sets to either E, H , G if thermodynamic data or E SCF if not
         min: str [optional, default=None] - index of moelcule to be treated as zero reference

        Returns:
         moleculeData: pandas DataFrame - dataframe of the molecules to plot with relative (E SCF)/(E/G/H) columns for plotting
    '''

    # Subset amount of data frame to plot
    if molsToPlot != None:
        moleculeData = moleculeDataFull.reindex(molsToPlot)
    else:
        moleculeData = moleculeDataFull

    # Calculate relative and normalised quantities
    if quantities == None:
        if 'G' in list(moleculeData.columns):
            quantities = ['E', 'H', 'G']
        else:
            quantities = ['E SCF']
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


def constructReactionPath(system_file, mol_names=None):

    # Read in system file
    with open(system_file) as file:
        input = file.read().splitlines()

    # Parse mol_names from system file if not already created
    if mol_names == None:
        mol_names = []
        for line in input:
            if line[0] != '#':
                mol_names.append(line.split()[0])

    # Set neighbour list from system file
    # Might not need branches, numSteps or even stepNeighbours
    branches = 1
    numSteps = 0
    stepNeighbours = []
    for line in input:
        if line[0] != '#':
            if len(line.split()) > 2:
                stepNeighbours.append(line.split()[2].split(','))
                branches += len(stepNeighbours[-1]) - 1
            else:
                stepNeighbours.append([])
            numSteps += 1

    # Set adjacency matrix
    adjacency = np.zeros((numSteps, numSteps))
    for node, edgeSet in enumerate(stepNeighbours):
        for edge in edgeSet:
            adjacency[node, mol_names.index(edge)] = 1

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
