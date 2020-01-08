import numpy as np
import sys


'''Module containing functions to calculate geometric parameters of a molecule'''


def calcParam(paramSet, geometry):

    '''Function which calculates the bond, valence angle or dihedral of the inputted parameters - can take multiple parameters

    Parameters:
     paramSet: Nested list of Ints - the sets of parameters to calculate the geometry for
     geometry: Numpy array (dim: numAtoms, 3) (float) - Results array of x, y, z coordinates for each atom (might work as list)

    Returns:
     paramVal: List of floats - the calculated parameters for each one in the inputted paramSet
    '''

    paramVal = []

    # Tests to see if multiple parameters or not
    if type(paramSet[0]) != list:
        paramSet = [paramSet]

    # Check number of indexes specifying each parameter and calcualte corresponding value (bond, angle or dihedral)
    for param in paramSet:
        if len(param) == 2:
            paramVal.append(calcDist(geometry[param[0]], geometry[param[1]]))
        elif len(param) == 3:
            paramVal.append(calcAngle(geometry[param[0]], geometry[param[1]], geometry[param[2]]))
        else:
            paramVal.append(calcDihedral(geometry[param[0]], geometry[param[1]], geometry[param[2]], geometry[param[3]]))
    return(paramVal)


def calcDist(atomOne, atomTwo):

    ''' Simple function whih calculates the distance between two atoms
        Parameters:
         atomOne - np array; x, y, z coordinates of atom one
         atomTwo - np array; x, y, z coordinates of atom two

        Returns:
         dist - float; distance between the two atoms
        '''
    # Calculates the bond vector between the two atoms
    bVec = atomOne - atomTwo
    # Calculates the inner product of the vectors (magnitude)
    dist = np.sqrt(np.dot(bVec, bVec))
    return dist


def calcAngle(atomOne, atomTwo, atomThree):

    ''' Simple function which calculates the angle between three atoms, middle atom is atomTwo
        Parameters:
         atomOne - np array; x, y, z coordinates of atom one
         atomTwo - np array; x, y, z coordinates of atom two
         atomThree - np array; x, y, z coordinates of atom three

        Returns:
         angle - float; angle between the two vectors: (atomTwo, atomOne) and (atomTwo, atomThree)
        '''
    # Calculate the two bond vectors
    bOneVec = atomOne - atomTwo
    bTwoVec = atomThree - atomTwo

    # Calculate the inner products of the two bonds with themselves and each other
    bOne = np.sqrt(np.dot(bOneVec, bOneVec))
    bTwo = np.sqrt(np.dot(bTwoVec, bTwoVec))
    angle = np.dot(bOneVec, bTwoVec)/(bOne*bTwo)

    # Return the angle between the bonds in degrees
    return np.arccos(angle)*(180/np.pi)


def calcDihed(atomOne, atomTwo, atomThree, atomFour):

    ''' Simple function to calculate the dihedral angle between four atoms
    Parameters:
     atomOne - np array; x, y, z coordinates of atom one
     atomTwo - np array; x, y, z coordinates of atom two
     atomThree - np array; x, y, z coordinates of atom three
     atomFour - np array; x, y, z coordinates of atom four

    Returns:
     dihedral - float; dihedral angle between the planes: (atomOne, Two, Three) and (atomTwo, Three, Four)
    '''

    bOneVec = atomTwo - atomOne
    bTwoVec = atomThree - atomTwo
    bThreeVec = atomFour - atomThree

    # Calculate the norms to the planes
    nOne = np.cross(bOneVec, bTwoVec)
    nTwo = np.cross(bTwoVec, bThreeVec)

    # Normalise the two vectors
    nOne /= np.linalg.norm(nOne)
    nTwo /= np.linalg.norm(nTwo)
    bTwoVec /= np.linalg.norm(bTwoVec)

    # Find third vector to create orthonormal frame
    mOne = np.cross(nOne, bTwoVec)

    # Evaluate n2 w.r.t the orthonormal basis
    x = np.dot(nTwo, nOne)
    y = np.dot(nTwo, mOne)

    return(np.arctan2(-y, x)*(180/np.pi))


def geomPullxyz(inputFile):

    '''Function which extracts the optimised geometry of a molecule from an .xyz file.

        Parameters:
         inputFile: Str - name of the input log file

        Returns:
         molCoords: Numpy array (dim: numAtoms, 3) (float) - Results array of x, y, z coordinates for each atom
        '''

    # Open and read input file
    with open(inputFile, 'r') as xyzFile:

        for el in xyzFile:

            # Set atom number from first line of xyz file
            numAtoms = int(el.strip())
            [xyzFile.__next__() for i in range(1)]

            molCoords = np.zeros((numAtoms, 3))
            atomIDs = []

            # Read in the atomic coordinates, atom ID will be row index
            for ind in range(numAtoms):
                el = xyzFile.__next__()
                atomIDs.append(str(el.split()[0]))
                for jind in range(1, 3):
                    molCoords[ind, jind] = float(el.split()[jind])

    return(molCoords, atomIDs)


def geomPushxyz(outputFile, numAtoms, atomIDs, coordinates):

    '''Function which outputs the extracted geometry to an .xyz file.

        Parameters:
         outputFile: Str - name of the output xyz file
    '''

    # Open output file, print header lines then atom indexes and cartesian coordinates to file
    with open(outputFile + '.xyz', 'w+') as output:
        print(numAtoms, file=output)
        print('Structure of {}'.format(outputFile.split('.')[0]), file=output)
        for atomInd, atom in enumerate(atomIDs):
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format(atom, coordinates[atomInd]), file=output)
