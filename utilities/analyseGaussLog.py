import numpy as np
import pandas as pd

'''
Module containing parsing functions to get information from gaussian log files.
'''

def countAtoms(logFile):

    '''Function to count the number of atoms in a molecule from a gaussian log file

    Returns:
     numAtoms: Int - The number of atoms in the system
    '''

    # Opens file and searches for line which contains the number of atoms in the system
    with open(logFile, 'r') as input:
        for el in input:
            if 'NAtoms' in el:
                numAtoms = int(el.split()[1])
                break

    return numAtoms


def pullGeom(logFile, optStep=1, SPE=False):

    '''Function that pulls the geometry from a gaussian log file

        Parameters:
         logFile: str - name of the input log file
         optStep: int - the target geometry to pull; useful in scans when specific step needed or when taking a geometry from earlier in an optimisation trajectory. Default = 1 (final geometry for Opt/Freq calcualions
         SPE: bool - flag for if SPE calcuation or rigid scan and no opt information in log

        Returns:
         atomCoords: ndarray (dim: numAtoms, 3; float) - Results array of x, y, z coordinates for each atom in the molecule
         optimsied: bool - flags whether the structure is optimised or not
    '''

    # Set count for number of 'optimized steps' encountered through file
    # How to set numAtoms or to just get from file?
    optCount = 0
    optimised = False
    numAtoms = countAtoms(logFile)
    atomCoords = np.zeros((numAtoms, 3))

    with open(logFile, 'r') as input:
        for el in input:

            if 'Standard orientation:' in el:
                # Skip the header section of the standard orientation block
                [input.__next__() for x in range(0,4)]
                # Read in the atomic coordinates, atom ID will be row index
                for atom in range(numAtoms):
                    el = input.__next__()
                    for jind in range(3):
                        atomCoords[atom, jind] = float(el.split()[jind+3])
                # If single point calculation then cancel on target standard orientation
                if SPE == True:
                    optCount += 1

            # Increments optCount if 'Optimized' met, breaks loop if target opt step is reached
            if 'Optimized Parameters' in el:
                optCount += 1
                if 'Non-Optimized' in el:
                    optimised = False
                else:
                    optimised = True
            if (optCount == optStep):
                return atomCoords, optimised

    return atomCoords, optimised


def pullEnergy(logFile, optStep=1, mp2=False, SPE=False):

    '''Function that pulls the energy from a gaussian log file

        Parameters:
         logFile: str - name of the input log file
         optStep: int - the target geometry to pull; useful in scans when specific step needed or when taking a geometry from earlier in an optimisation trajectory. Default = 1 (final geometry for Opt/Freq calcualions
         mp2: bool - flag to search for the MP2 energy instead of the HF SCF if MP2 calculation

        Returns:
         eSCF: float - SCF Done energy in a.u.
         optimsied: bool - flags whether the structure is optimised or not
    '''

    optCount = 0
    optimised = False
    with open(logFile, 'r') as input:
        for el in input:

            # NB: SCF Done precede the corresponding optimisation section
            if 'SCF Done:' in el:
                eSCF = float(el.split('=')[1].split()[0])
            # MP2 energy printed out seperately - has to be processed to float form
            if mp2 == True:
                if 'EUMP2' in el:
                    mp2Raw = el.split('=')[2].strip()
                    eSCF = float(mp2Raw.split('D')[0])*np.power(10, float(mp2Raw.split('D')[1]))

            # Increments optCount if 'Optimized' met, breaks loop if target opt step is reached
            if 'Optimized Parameters' in el:
                optCount += 1
                if 'Non-Optimized' in el:
                    optimised = False
                else:
                    optimised = True
                if (optCount == optStep):
                    return eSCF, optimised

            # If single point calculation then cancel on target standard orientation (follows SCF output)
            if 'Standard orientation' in el:
                if SPE == True:
                    optCount += 1
                    if (optCount == optStep+1):
                        return eSCF, optimised

    return eSCF, optimised


def pullThermo(logFile, eSCF=0):

    '''Function which pulls thermodynamic information from a gaussian Freq log file

        Parameters:
         logFile - str; file name of the .log file to be read

        Returns:
         list of floats:
            totE - thermally corrected energy (kJ/mol)
            totH - enthalpy (kJ/mol)
            totG - free energy (kJ/mol)
            totS - entropy (kJ/mol)
            zpe - zero point energy (kJ/mol)
    '''

    if eSCF == 0:
        eSCF = pullEnergy(logFile)[0]

    # Open the file
    with open(logFile) as input:
        # Search file for the thermodynamic data
        for line in input:
            if 'Temperature ' in line:
                temp = float(line[15:22])
            # Start the thermochemistry section, save  ZPE and thermal corrections for E, H and G
            if 'Zero-point correction= ' in line:
                zpe = float(line[50:58])
                tCorrE = float(next(input)[50:58])
                tCorrH = float(next(input)[50:58])
                tCorrG = float(next(input)[50:58])

    # Calculate the total entropy (S) and thermally corrected E, G and H
    totS = (tCorrH - tCorrG)/temp
    totE = eSCF + tCorrE
    totH = eSCF + tCorrH
    totG = eSCF + tCorrG

    # Convert to kJ/mol
    return [value*2625.5 for value in [totE, totH, totG, totS, zpe]]


def pullAtomIDs(logFile):

    '''Function that pulls the atom IDs from a gaussian log file

        Parameters:
         logFile: str - name of the input log file

        Returns:
         atomIDs: list of str - atom IDs of the atoms in the molecule
    '''

    atomIDs = []
    numAtoms = countAtoms(logFile)

    with open(logFile, 'r') as input:
        for el in input:

            # Sets atomIDs from initialising list of input structure
            if 'Charge = ' in el:
                el = input.__next__()
                if ('No Z-Matrix' in el) or ('Redundant internal coordinates' in el):
                    el = input.__next__()
                for atom in range(numAtoms):
                    atomIDs.append(el.split()[0][0])
                    el = input.__next__()
                break

    return atomIDs


def pullScanInfo(logFile):

    '''Function that pulls information about a scan from a gaussian log file

         Parameters:
          logFile: str - name of the input log file

         Returns:
          scanInfo: dict -
              {paramKey: str,
                atomInd: list of ints,
                nSteps: int,
                stepSize: float}
        Where, paramKey  is a string identifier of the modRed parameter type
        NB: atomInd is pythonic
     '''

    # Types dictionary of the corresponding number of atom IDs required for each one
    types = {'X': 1, 'B': 2, 'A': 3, 'D': 4}
    modRedundant = []

    # Opens and reads file, extracting the modRedundant input to a list (each modRed input line new object in list)
    with open(logFile, 'r') as input:
        for el in input:
            if 'The following ModRedundant input section has been read:' in el:
                el = input.__next__()
                # Extracts the ModRedundant section
                while el.strip() != '':
                    modRedundant.append(el.strip().split())
                    el = input.__next__()
                break

    # Gets the atom IDs for setting the parameter key
    atomIDs = pullAtomIDs(logFile)

    # Iterates over the modRedundant inputs, finds the scan parameter and saves the input
    for mR in modRedundant:
        # Identifies number of atom IDs to expect and tests the action input for the scan parameter (assuming only one here, could have more)
        numAtomsParam = types[mR[0]]
        if mR[numAtomsParam+1] == 'S':
            scanInfo = {'paramKey': '', 'atomInd': [], 'nSteps': int(mR[-2]), 'stepSize': float(mR[-1])}
            # Set paramKey and atom indexes (NB: deduct 1 for python indexing) of scan parameter
            for atomInd in mR[1:numAtomsParam]:
                scanInfo['atomInd'].append(int(atomInd) - 1)
                scanInfo['paramKey'] += (atomIDs[int(atomInd) - 1] + atomInd + '-')
            scanInfo['atomInd'].append(int(mR[numAtomsParam]) - 1)
            scanInfo['paramKey'] += (atomIDs[int(mR[numAtomsParam]) - 1] + mR[numAtomsParam])

    try:
        return scanInfo
    except NameError:
        print('No scan parameter located')
        raise


def pullRigidScanInfo(logFile):

    '''Function that pulls the variables and energy for each step in a gaussian rigid scan log file

        Parameters:
         logFile: str - name of the input log file

        Returns:
         eSCF: float - SCF Done energy in a.u.
         optimsied: bool - flags whether the structure is optimised or not
    '''

    # Initialise variables
    scanVars = [] # Make a list of dict with entries to match the details (name, starting value, steps, step size')
    scanPoints = 1
    initialzMat = []
    scanVariables = []

    # Parse log file for variables
    with open(logFile, 'r') as input:
        for el in input:

            # Pulls initial z matrix
            if 'Charge' in el:
                el = input.__next__()
                while 'Variables:' not in el:
                    initialzMat.append(el.strip())
                    el = input.__next__()

            # Set dicts of scan variables
            if 'Variables:' in el:
                el = input.__next__()
                while el.strip() != '':
                    varInput = el.split()
                    scanVariables.append(varInput[0])
                    currentScanVar = {'Name': varInput[0], 'Initial value': float(varInput[1][:-4]), 'Steps': int(varInput[2]), 'Step size' : float(varInput[3])}
                    scanVars.append(currentScanVar)
                    scanPoints *= (currentScanVar['Steps'] + 1)
                    el = input.__next__()
                break

    return scanVariables, scanPoints, initialzMat


#def pullRigidScan(logFile):
#
#    '''Function that pulls the values of parameters and the energy for each step in a gaussian rigid scan log file
#
#        Parameters:
#         logFile: str - name of the input log file
#         mp2: bool - flag to search for the MP2 energy instead of the HF SCF if MP2 calculation
#
#        Returns:
#         eSCF: float - SCF Done energy in a.u.
#         optimsied: bool - flags whether the structure is optimised or not
#    '''
#
#    # Initialise variables
#
#    # Parse log file for variables
#    with open(logFile, 'r') as input:
#        for el in input:
#
#
#
#                # Set results array for variable results - columns are variables and
#                scanResults = np.zeros((scanPoints, len(
#
#            # SCF Done precede the variable value output for the step
#            if 'SCF Done:' in el:
#                eSCF = float(el.split('=')[1].split()[0])
#            # MP2 energy printed out seperately - has to be processed to float form
#            if mp2 == True:
#                if 'EUMP2' in el:
#                    mp2Raw = el.split('=')[2].strip()
#                    eSCF = float(mp2Raw.split('D')[0])*np.power(10, float(mp2Raw.split('D')[1]))
#
#            if 'Variable ' in el:
#                el = input.__next__()

