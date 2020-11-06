"""Module containing functions to calculate geometric parameters."""

import sys
import numpy as np

def calc_param(param_set, geometry):
    """
    Calculate geometric parameters (bond distance, angle or dihedral).
    
    Multiple geometric parameters can be calculated for an input geometry. 
    Geometric parameter calculated is defined by number of atom indexes.
    Example:
        [0, 1]: Bond distance
        [0, 1, 2]: Bond angle
        [0, 1, 2, 3]: Dihedral  

    Parameters
    ----------
    param_set : nested list of ints
        Atom indexes for each parameter to be calculated.
    geometry : `numpy.ndarray`
        A `(numAtoms, 3)` array of x, y, z coordinates for each atom.

    Returns
    -------
    param_val : `list of floats`
        Calculated parameter values.
    
    """
    # Initialise variables.
    param_val = []

    # Checks for multiple paramters.
    if type(param_set[0]) != list:
        param_set = [param_set]

    # Calculate geometric paramter for the number of atom indexes specified.
    for param in param_set:
        if len(param) == 2:
            param_val.append(calc_dist(geometry[param[0]], geometry[param[1]]))
        elif len(param) == 3:
            param_val.append(calc_angle(geometry[param[0]], geometry[param[1]], geometry[param[2]]))
        else:
            param_val.append(calc_dihed(geometry[param[0]], geometry[param[1]], geometry[param[2]], geometry[param[3]]))
    return(param_val)


def calc_dist(atom_one, atom_two):
    """
    Calculate bond distance between two atoms.

    Parameters
    ----------
    atom_one : `numpy.ndarray`
        A `(1, 3)` array of x, y, z coordinates of atom one.
    atom_two : `numpy.ndarray`
        A `(1, 3)` array of x, y, z coordinates of atom two.

    Returns
    -------
    dist : `float`
        Distance between two atoms.

    """
    # Calculates the bond vector between the two atoms
    b_vec = atom_one - atom_two
    # Calculates the inner product of the vectors (magnitude)
    dist = np.sqrt(np.dot(b_vec, b_vec))
    return dist


def calc_angle(atom_one, atom_two, atom_three):
    """
    Calculate angle between three atoms (A(a, b, c) for atoms bonded: a-b-c).

    Parameters
    ----------
    atom_one : `numpy.ndarray`
        A `(1, 3)` array of x, y, z coordinates of atom one.
    atom_two : `numpy.ndarray`
        A `(1, 3)` array of x, y, z coordinates of atom two.
    atom_three : `numpy.ndarray`
        A `(1, 3)` array of x, y, z coordinates of atom three.

    Returns
    -------    
    angle : `float`
        Angle between three atoms.

    """
    # Calculate the two bond vectors (0-1 and 1-2)
    b_one_vec = atom_one - atom_two
    b_two_vec = atom_three - atom_two

    # Calculate the inner products of the two bonds with themselves and each other
    b_one = np.sqrt(np.dot(b_one_vec, b_one_vec))
    b_two = np.sqrt(np.dot(b_two_vec, b_two_vec))
    angle = np.dot(b_one_vec, b_two_vec)/(b_one*b_two)

    # Return the angle between the bonds in degrees
    return np.arccos(angle)*(180/np.pi)


def calc_dihed(atom_one, atom_two, atom_three, atom_four):
    """
    Calculate dihedral between four atoms (D(a,b,c,d) for atoms: a-b-c-d).

    Parameters
    ----------
    atom_one : `numpy.ndarray`
        A `(1, 3)` array of x, y, z coordinates of atom one.
    atom_two : `numpy.ndarray`
        A `(1, 3)` array of x, y, z coordinates of atom two.
    atom_three : `numpy.ndarray`
        A `(1, 3)` array of x, y, z coordinates of atom three.
    atom_four : `numpy.ndarray`
        A `(1, 3)` array of x, y, z coordinates of atom four.

    Returns
    -------
    dihedral : `float`
        Dihedral between four atoms.

    """
    # Calculate bond vectors.
    b_one_vec = atom_two - atom_one
    b_two_vec = atom_three - atom_two
    b_three_vec = atom_four - atom_three

    # Calculate norms to the planes.
    n_one = np.cross(b_one_vec, b_two_vec)
    n_two = np.cross(b_two_vec, b_three_vec)

    # Normalise the vectors.
    n_one /= np.linalg.norm(n_one)
    n_two /= np.linalg.norm(n_two)
    b_two_vec /= np.linalg.norm(b_two_vec)

    # Find third vector to create orthonormal frame.
    m_one = np.cross(n_one, b_two_vec)

    # Evaluate n2 w.r.t the orthonormal basis.
    x = np.dot(n_two, n_one)
    y = np.dot(n_two, m_one)

    return np.arctan2(-y, x)*(180/np.pi)


def geom_pull_xyz(input_file):
    """
    Pull geometry of a molecule from an .xyz file.

    Parameters
    ----------
    input_file : `str`
        Name/path of the input xyz file/

    Returns
    -------
    mol_coords : `numpy.ndarray`
        A `(numAtoms, 3)` of x, y, z coordinates for each atom
    
    """
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

    return mol_coords, atom_ids


def geom_push_xyz(output_file, num_atoms, atom_ids, coordinates):
    """
    Write geometry to an .xyz file.

    Parameters
    ----------
    output_file : `str`
        Name/path of output xyz file.

    """
    # Open output file, print header lines then atom indexes and cartesian coordinates to file
    with open(output_file + '.xyz', 'w+') as output:
        print(num_atoms, file=output)
        print('Structure of {}'.format(output_file.split('.')[0]), file=output)
        for atom_ind, atom in enumerate(atom_ids):
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format(atom, coordinates[atom_ind]), file=output)

