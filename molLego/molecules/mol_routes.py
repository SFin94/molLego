"""Module of functions that use Molecule objects."""
import sys
import numpy as np
import pandas as pd
import itertools

import molLego.parsers.parse_gaussian as pgauss
import molLego.utilities.geom as geom
from molLego.molecules.molecule import Molecule

def construct_mols(system_file, parser, molecule_type=Molecule):
    """
    Create Molecules for output files defined by a system conf file.

    The .conf file contains molecule names and files to be parsed.
    Multiple files can be parsed for one molecule name.
    Example formatting:
        molecule_1_name molecule_1_output[.ext]
        molecule_2_name molecule_2a_output[.ext],molecule_2b_output[.ext]
        # molecule_3_name molecule_3_output[.ext]

    Where [.ext] must be compatiable with the parser specified.
    Multiple output files are csv (molecule_2).
    Lines can be commented out with leading '#' (molecule_3).

    Parameters
    ----------
    system_file : `str`
        File path/name to conf file containing system to parse.

    parser : `OutputParser`
        Parser class to use for calculation output.

    molecule_type : `Molecule`
        Molecule class to use for calculation output.

    Returns
    -------
    molecules : `dict` of :Molecule:
        Molecule objects for each file in system conf file.

    """
    # Initialise variables
    mol_names = []
    molecules = []

    # Process files and names in system conf file.
    with open(system_file, 'r') as infile:
        for system_line in infile:
            if system_line[0] != '#':
                # Set name and files from input line.
                mol_files = system_line.split()[1].split(',')
                
                # Create molecules for each file.
                for mol in mol_files:
                    mol_names.append(system_line.split()[0])
                    molecules.append(molecule_type(output_file=mol, parser=parser))

    return mol_names, molecules

def mols_to_dataframe(mols, mol_names=None, 
                      save=None, mol_zero=None):
    """
    Create DataFrame of Molecules with relative values.

    Parameters
    ----------
    mols : `list of :Molecule:`
        Molecules to send to dataframe.
    
    mol_names : `list of str` 
        [Default=None]
        If ``None`` then DataFrame index is Molecule file name.

    save : `str`
        [Default=None].
        File name to write DataFrame to (w/out .csv).
        If ``None`` then DataFrame is not written to file.

    mol_zero : `str` or `int`
        [Default=None]
        Molecule to calculate values relative too. 
        Can be `str` of mol_name of Molecule 
        Or `int` index of Molecule in mols list.
        If ``None`` relative values calculated w.r.t. lowest
        value for each quantity.

    Returns
    -------
    molecule_df : :pandas: `DataFrame`
        DataFrame of Molecules and properties.
     
    """
    # Create data frame representations.
    mol_data = [mol.get_df_repr() for mol in mols]
    
    # Handle possible nested dicts.
    if not isinstance(mol_data[0], dict):
        mol_data = itertools.chain(*mol_data)
    
    # Create dataframe and calculate relative values.
    molecule_df = pd.DataFrame(mol_data, index=mol_names)
    if isinstance(mol_zero, int):
        mol_zero = mol_names[mol_zero]
    molecule_df = calc_relative(molecule_df, mol_zero=mol_zero)

    # Write dataframe to file if filename provided.
    if save != None:
        molecule_df.to_csv(save + '.csv')

    return molecule_df

def calc_relative(molecule_df, quantities=None, mol_zero=None):
    """
    Calculate relative values in Molecule DataFrame.
    
    Parameters
    ----------
    molecule_df : :pandas: `DataFrame`
        DataFrame of molecule properties.

    quantities: `list of str`
        [Default=None] 
        The quantitity/ies to calculate relative
        values for (str should match DataFrame heading).
        If ``None`` default to e or e/h/g depending on dataframe. 
    
    mol_zero : `str``
        [Default=None]
        Index of molecule to calculate values relative too.
        If ``None`` relative values calculated w.r.t. lowest
        value for each quantity.
    
    Returns
    -------
    molecule_df : :pandas: `DataFrame`
        Updated DataFrame of relative molecule properties.

    """
    # Set quantities to those present in dataframe is None given.
    if quantities == None:
        all_quantities = ['e', 'h', 'g']
        present = np.asarray([x in list(molecule_df.columns)
                            for x in all_quantities])
        quantities = [all_quantities[x] for x in np.where(present)[0]]

    # Find zero value for quantities and set other values relative.
    for q in quantities:
        if mol_zero != None:
            zero = molecule_df[q, mol_zero]
        else:
            zero = molecule_df[q].min()
        molecule_df['relative '+q] = molecule_df[q] - zero

    return molecule_df

def parse_tracked_params(param_file, molecules=None):
    """
    Parse paramaters from input file and calculate values.

    Format of input file:
            param_name (atom_types) atom1_ind atom2_ind [atom3_ind atom4_ind]
            E.g. OPSC 3 1 2 7

    Parameters
    ----------
    param_file : :class:`str`
        Path of input file containg parameters to be calculated.
    
        Format of input file:
             param_name (atom_types) atom1_ind atom2_ind [atom3_ind atom4_ind]
             E.g. OPSC 3 1 2 7

    molecules: :class:`list` of :Molecule:
        The molecules to calcualte the parameters values for.
        [Default=None]
       
    Returns
    -------
    tracked_param : :class:`dict`
        Key is the param_name from the file and
        Value is the atom indexes (0 index) that
        define the atoms involved in the parameter.

    """
    # Initialise empty dict for params
    tracked_params = {}
    # Parse in file and seperate the indexes from the parameter ID and save as an entry to the dict.
    with open(param_file, 'r') as infile:
        for el in infile:
            param = el.strip().split(' ')
            indexes = [int(ind)-1 for ind in param[1:]]
            tracked_params[param[0]] = indexes
    
    # Calculate parameter values for each molecule.
    if molecules is not None:
        for mol in molecules:
            mol.set_parameters(tracked_params)
    
    return tracked_params
