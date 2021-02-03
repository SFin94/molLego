"""Module containing reaction profile functions."""
import sys
import numpy as np
import pandas as pd

import molLego.molecules.mol_routes as molroutes

def track_reaction_path(current_step, adjacency, current_path=[]):
    """
    Construct a branch of a reaction path.

    Parameters
    ----------
    current_step: `int`
        Index of molecule of the current step in pathway.
    adjacency: :numpy:`array`
        Connecitivty matrix.
        Entries are 1 for connected points or 0 for unconnected points.
    current_path: `list`
        Pathway up to current reaction step.

    Returns
    -------
    path_list: `nested list`
        List of reaction steps for each reaction path.
    
    """
    # Append step to path and finish if final point.
    current_path = current_path + [current_step]
    if np.count_nonzero(adjacency[current_step,:]) == 0:
        return current_path

    # Locate connecting reaction steps for current paths.
    paths = []
    next_path = np.nonzero(adjacency[current_step,:])[0]
    for path in next_path:
        next_step = track_reaction_path(path, adjacency, current_path)
        for step in next_step:
            paths.append(step)
    return paths

def construct_reaction_path(conf_file):
    """
    Construct reaction paths from a reaction .conf file.

    Parameters
    ----------
    conf_file : `str`
        Reaction .conf file

    Returns
    -------
    path_list : `nested list`
        List of reaction steps for each reaction path.

    """
    # Read in reaction .conf file
    with open(conf_file) as infile:
        reaction_input = infile.read().splitlines()

    # Parse mol_names from .conf file input.
    mol_names = []
    for el in reaction_input:
        if el[0] != '#':
            mol_names.append(el.split()[0])

    # Set neighbour list.
    branches = 1
    num_steps = 0
    step_neighbours = []
    for el in reaction_input:
        if el[0] != '#':
            if len(el.split()) > 2:
                step_neighbours.append(el.split()[2].split(','))
                branches += len(step_neighbours[-1]) - 1
            else:
                step_neighbours.append([])
            num_steps += 1

    # Set adjacency matrix.
    adjacency = np.zeros((num_steps, num_steps))
    for node, edge_set in enumerate(step_neighbours):
        for edge in edge_set:
            adjacency[node, mol_names.index(edge)] = 1

    # Calculate path list from adjacency.
    path_list = []
    reactant_nodes = np.nonzero(np.sum(adjacency, axis=0) == 0)[0]
    for r_node in reactant_nodes:
        path_list.append(track_reaction_path(r_node, adjacency))

    # Convert index list to list of molecule names.
    reaction_paths = []
    for path in path_list:
        reactant_node = path[0]
        path_mols = [mol_names[reactant_node]]
        for rstep in path[1:]:
            # Split to new path if reactants.
            if rstep == reactant_node:
                reaction_paths.append(path_mols)
                path_mols = [mol_names[reactant_node]]
            else:
                path_mols.append(mol_names[rstep])
        reaction_paths.append(path_mols)
    
    return reaction_paths

def construct_reaction_profile(reaction_steps_data, reaction_paths, 
                               path_min=None, quantity=None, save=None):
    """
    Construct reaction profile data for reaction paths.

    Parameters
    ----------
    reaction_steps_data : :pandas:`DataFrame`
        Molecule data for all unordered reaction steps involved in 
        reaction profile.
    reaction_paths: `nested lists`
        List of reaction steps for each reaction path.
    path_min: `list`
        Reaction step to be taken as 'zero' for relative thermodynamic 
        quantities. One for each path. 
        [default: None] If None, uses lowest valued reaction step.

    Returns
    -------
    reaction_profile_data: :pandas:`DataFrame`
        Molecule data and realative quantities for each reaction path 
        in reaction profile.

    """
    reaction_profile_data = pd.DataFrame()
    # Set no path minimum for each reaction path.
    if path_min == None:
        path_min = [None]*len(reaction_paths)

    for i, rpath in enumerate(reaction_paths):

        # Calculate reaction coordinates.
        reac_coord = np.linspace(0, 1, len(reaction_paths))
        
        # Subset dataframe with path names.
        rpath_data = reaction_steps_data.loc[rpath].copy()
        rpath_data = molroutes.calc_relative(rpath_data, quantities=quantity, 
                                             min=path_min[i])
        
        # Add reaction coordinate and path number as column.
        rpath_data['Reaction path'] = i
        rpath_data['Rx'] = np.linspace(0, 1, len(rpath))
        
        # Append reaction path to overall reaction profile.
        reaction_profile_data = reaction_profile_data.append(rpath_data)

    # Write dataframe to file if filename provided.
    if save != None:
        reaction_profile_data.to_csv(save + '.csv')

    return reaction_profile_data


