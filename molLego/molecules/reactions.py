"""Module containing definition of a reaction."""
import sys
import numpy as np

from molLego.molecules.molecule import Molecule
from molLego.parsers.parse_combine import CombineParser

class Reaction():
    """
    Represents a Reaction with Molecules as reaction steps.

    Attributes
    ----------
    num_steps : :class:`int`
        Number of different reaction steps in the reaction.

    reaction_steps : :class:`list` of :Molecule:
        The Molecules composing the reaction steps.
        Where multiple molecules form a single reaction step
        the should be contained in a nested list.
        
    """

    def __init__(self, reaction_steps, neighbours):
        """
        Initialise a Molecule from calculation output file.

        Parameters
        ----------
        reaction_steps : :class:`list` of :Molecule:
            The Molecules composing the reaction steps.
            Where multiple molecules form a single reaction step
            the should be contained in a nested list.

        neighbours : nested :class:`list` of :class:`int`
            A list of indexes of the upstream connecting reaction 
            steps for each reaction step. 
            Products/terminating reaction steps will have an 
            empty value. 

        """
        self.num_steps = len(reaction_steps)

        # Initialise reaction steps from molecules.
        self.reaction_steps= []
        for step in reaction_steps:
            
            # Handle is multiple molecules for single reaction step.
            if isinstance(step, (list,tuple)):
                print(step)
                self.reaction_steps.append(self.form_reaction_step(
                                           step))
            else:
                self.reaction_steps.append(step)
        
        self.reaction_paths = self.construct_reaction_paths(neighbours)
        self.num_paths = len(self.reaction_paths)
            
    def get_reaction_path(self, path_index=None):
        """
        Yield reaction path in the reaction.

        Parameters
        ----------
        path_index : :class: `iterable` of :class:`int`
            The index(es) of the reaction paths required.
            [default: ``None``] If ``None`` then returns all.
            Can be single `int` to call name for single reaction path.

        Yields
        -------
        :class: `list`
            List of Molecule objects making up the reaction path.

        """
        # Set to all atoms is atom_index is None.
        if path_index is None:
            path_index = range(len(self.reaction_paths))
        elif isinstance(path_index, int):
            path_index = (path_index, )

        paths = [self.reaction_paths[i] for i in path_index]

        for path in paths:
            yield path

    def get_df_repr(self, path_index=None):
        """
        Yield dict representation of reaction paths in reaction.

        Parameters
        ----------
        path_index : :class: `iterable` of :class:`int`
            The index(es) of the reaction paths required.
            [default: ``None``] If ``None`` then returns all.
            Can be single `int` to call name for single reaction path.

        Yields
        -------
        df_rep : nested :class:`dict`
            Properties for each reaction step in reaction path. Nested dict where each reaction step 
            entry has the form:
            {
                file_name : path to parent output file
                reaction path :  reaction path number
                rx : reaction coordinate
                e : energy (thermally corrected if known) (kJ/mol)
                h/g/s : additional quantities if Molecule attributes
                parameter key : parameter value
                [for all parameters in self.parameters]
            }

        """
        # Set to all atoms is atom_index is None.
        if path_index is None:
            path_index = range(len(self.reaction_paths))
        elif isinstance(path_index, int):
            atom_index = (path_index, )

        paths = (self.reaction_paths[x] for x in path_index)
        
        for i, path in enumerate(paths):
            df_rep = {}
            rx = np.linspace(0, 1, len(path))
        
            # Get df rep for each reaction step.
            for j, step in enumerate(path):
                step_df_rep = step.get_df_repr()
                step_df_rep['reaction path'] = path_index[i]
                df_rep[rx[j]] = step_df_rep
            
            yield df_rep

    def _track_reaction_path(self, 
                             current_step, 
                             adjacency, 
                             current_path=[]):
        """
        Construct a branch of a reaction path.

        Parameters
        ----------
        current_step : :class:`int`
            Index of molecule of the current step in pathway.
        
        adjacency : :class:`numpy ndarray`
            A ``(N, N)`` array showing connectivity between reaction
            steps. Values of 1 show connected points and 0 show unconnected points.
        
        current_path : :class:`list`
            Pathway up to current reaction step.

        Returns
        -------
        paths : nested :class:`list` of :class:`int`
            Reaction paths defined by indexes of reaction steps.
        
        """
        # Append step to path and finish if final point.
        current_path = current_path + [current_step]
        if np.count_nonzero(adjacency[current_step,:]) == 0:
            return current_path

        # Locate connecting reaction steps for current paths.
        paths = []
        next_path = np.nonzero(adjacency[current_step,:])[0]
        for path in next_path:
            next_step = self._track_reaction_path(path, 
                                                 adjacency, 
                                                 current_path)
            for step in next_step:
                paths.append(step)
        return paths

    def construct_reaction_paths(self, neighbour_indexes):
        """
        Construct reaction paths made of the reaction step Molecules.

        Parameters
        ----------
        step_neighbours : :class:`list` of 
            The upstream connected reaction steps in terms of
            reaction step index.

        Returns
        -------
        reaction_paths : nested :class:`list` of :Molecule:
            Molecules along each connected reaction path in
            reaction system.

        """
        # Set adjacency matrix from neighbour list.    
        adjacency = np.zeros((self.num_steps, self.num_steps))
        for node, edge_set in enumerate(neighbour_indexes):
            
            # Handle if single int given.
            if isinstance(edge_set, int):
                edge_set = [edge_set]
            for edge in edge_set:
                adjacency[node, edge] = 1

        # Calculate path list from adjacency.
        path_list = []
        reactant_nodes = np.nonzero(np.sum(adjacency, axis=0) == 0)[0]
        for r_node in reactant_nodes:
            path_list.append(self._track_reaction_path(r_node,
                                                        adjacency))
        
        reaction_paths = []
        for path in path_list:
            reactant_node = path[0]

        # Convert index list to list of Molecules.
        reaction_path = []
        for path in path_list:
            reactant_node = path[0]
            path_steps = [self.reaction_steps[reactant_node]]
            for rstep in path[1:]:

                # Split to new path if reactants.
                if rstep == reactant_node:
                    reaction_paths.append(path_steps)
                    path_steps = [self.reaction_steps[reactant_node]]
                else:
                    path_steps.append(self.reaction_steps[rstep])
            reaction_paths.append(path_steps)

        return reaction_paths

    def add_reaction_path(self, path_list):
        """
        Add a new reaction path to the reaction.

        Parameters
        ----------
        path_list : :class:`list` of :class:`int`
            New reaction path in terms of the reaction step indexes.

        """
        reaction_path = []
        for step in path_list:
            reaction_path.append(self.reaction_steps[step])
    
        self.reaction_paths.append(reaction_path)
        self.num_paths += 1

    def form_reaction_step(self, molecules):
        """
        Sum energetic properties of molecules for single reaction step.

        Parameters
        ----------
        molecules : :class:`list` of :Molecule:
            Molecules to be combined for single reaction step.

        Returns
        -------
        :class:`Molecule`
            Dummy Molecule of the combined input molecules.

        """
        # Find Molecule type and create reaction step.
        mol_type = type(molecules[0])
        return mol_type(molecules, parser=CombineParser)



