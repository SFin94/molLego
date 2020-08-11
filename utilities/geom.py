import numpy as np
import sys


'''Module containing functions to calculate geometric parameters of a molecule'''


def calc_param(param_set, geometry):

    '''Function which calculates the bond, valence angle or dihedral of the inputted parameters - can take multiple parameters

    Parameters:
     param_set: nested list of ints - the sets of parameters to calculate the geometry for
     geometry: np array (dim: numAtoms, 3) (float) - Results array of x, y, z coordinates for each atom (might work as list)

    Returns:
     param_val: list of floats - the calculated parameters for each one in the inputted param_set
    '''

    param_val = []

    # Tests to see if multiple parameters or not
    if type(param_set[0]) != list:
        param_set = [param_set]

    # Check number of indexes specifying each parameter and calcualte corresponding value (bond, angle or dihedral)
    for param in param_set:
        if len(param) == 2:
            param_val.append(calc_dist(geometry[param[0]], geometry[param[1]]))
        elif len(param) == 3:
            param_val.append(calc_angle(geometry[param[0]], geometry[param[1]], geometry[param[2]]))
        else:
            param_val.append(calc_dihed(geometry[param[0]], geometry[param[1]], geometry[param[2]], geometry[param[3]]))
    return(param_val)


def calc_dist(atom_one, atom_two):

    ''' Simple function whih calculates the distance between two atoms
        Parameters:
         atom_one - np array; x, y, z coordinates of atom one
         atom_two - np array; x, y, z coordinates of atom two

        Returns:
         dist - float; distance between the two atoms
        '''
    # Calculates the bond vector between the two atoms
    b_vec = atom_one - atom_two
    # Calculates the inner product of the vectors (magnitude)
    dist = np.sqrt(np.dot(b_vec, b_vec))
    return dist


def calc_angle(atom_one, atom_two, atom_three):

    ''' Simple function which calculates the angle between three atoms, middle atom is atom_two
        Parameters:
         atom_one - np array; x, y, z coordinates of atom one
         atom_two - np array; x, y, z coordinates of atom two
         atom_three - np array; x, y, z coordinates of atom three

        Returns:
         angle - float; angle between the two vectors: (atom_two, atom_one) and (atom_two, atom_three)
        '''
    # Calculate the two bond vectors
    b_one_vec = atom_one - atom_two
    b_two_vec = atom_three - atom_two

    # Calculate the inner products of the two bonds with themselves and each other
    b_one = np.sqrt(np.dot(b_one_vec, b_one_vec))
    b_two = np.sqrt(np.dot(b_two_vec, b_two_vec))
    angle = np.dot(b_one_vec, b_two_vec)/(b_one*b_two)

    # Return the angle between the bonds in degrees
    return np.arccos(angle)*(180/np.pi)


def calc_dihed(atom_one, atom_two, atom_three, atom_four):

    ''' Simple function to calculate the dihedral angle between four atoms
    Parameters:
     atom_one - np array; x, y, z coordinates of atom one
     atom_two - np array; x, y, z coordinates of atom two
     atom_three - np array; x, y, z coordinates of atom three
     atomFour - np array; x, y, z coordinates of atom four

    Returns:
     dihedral - float; dihedral angle between the planes: (atom_one, Two, Three) and (atom_two, Three, Four)
    '''

    b_one_vec = atom_two - atom_one
    b_two_vec = atom_three - atom_two
    b_three_vec = atom_four - atom_three

    # Calculate the norms to the planes
    n_one = np.cross(b_one_vec, b_two_vec)
    n_two = np.cross(b_two_vec, b_three_vec)

    # Normalise the two vectors
    n_one /= np.linalg.norm(n_one)
    n_two /= np.linalg.norm(n_two)
    b_two_vec /= np.linalg.norm(b_two_vec)

    # Find third vector to create orthonormal frame
    m_one = np.cross(n_one, b_two_vec)

    # Evaluate n2 w.r.t the orthonormal basis
    x = np.dot(n_two, n_one)
    y = np.dot(n_two, m_one)

    return(np.arctan2(-y, x)*(180/np.pi))


def geom_pull_xyz(input_file):

    '''Function which extracts the optimised geometry of a molecule from an .xyz file.

        Parameters:
         input_file: Str - name of the input log file

        Returns:
         mol_coords: Numpy array (dim: numAtoms, 3) (float) - Results array of x, y, z coordinates for each atom
        '''

    # Open and read input file
    with open(input_file, 'r') as xyz_file:

        for el in xyz_file:

            # Set atom number from first line of xyz file
            num_atoms = int(el.strip())
            [xyz_file.__next__() for i in range(1)]

            mol_coords = np.zeros((num_atoms, 3))
            atom_ids = []

            # Read in the atomic coordinates, atom ID will be row index
            for i in range(num_atoms):
                el = xyz_file.__next__()
                atom_ids.append(str(el.split()[0]))
                for j in range(1, 3):
                    mol_coords[i, j] = float(el.split()[j])

    return(mol_coords, atom_ids)


def geom_push_xyz(output_file, num_atoms, atom_ids, coordinates):

    '''Function which outputs the extracted geometry to an .xyz file.

        Parameters:
         outputFile: Str - name of the output xyz file
    '''

    # Open output file, print header lines then atom indexes and cartesian coordinates to file
    with open(output_file + '.xyz', 'w+') as output:
        print(num_atoms, file=output)
        print('Structure of {}'.format(output_file.split('.')[0]), file=output)
        for atom_ind, atom in enumerate(atom_ids):
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format(atom, coordinates[atom_ind]), file=output)

