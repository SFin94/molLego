"""Module of functions that use Molecule objects."""
import sys
import numpy as np
import pandas as pd

import molLego.parsers.parse_gaussian as pgauss
import molLego.utilities.geom as geom
from molLego.molecules.molecule import Molecule

def construct_mols(system_file, parser, molecule_type=Molecule):
    """
    Create Molecules for output files defined by a system conf file.

    The conf file contains molecule names and files to be parsed.
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
                mol_names.append(system_line.split()[0])
                mol_files = system_line.split()[1].split(',')
                
                # Initialise molecules for each file.
                mols = [molecule_type(output_file=x, parser=parser) for x in mol_files]
                # Need to think about how to handle combining molecules.
                if len(mols) > 1:
                    """Currently won't function"""
                    molecules.append(sum_mols(mols))
                else:
                    molecules.append(mols[0])

    return mol_names, mols

def mols_to_dataframe(mols, mol_names=None, save=None, mol_zero=None):
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
    mol_data = []
    for i, mol in enumerate(mols):
        mol_data.append(mol.get_df_repr())

    # Set index from file name if names not given.
    if mol_names == None:
        mol_names = [mol.parser.file_name.split('/')[-1].split('.')[0]
                     for mol in mols]
    
    # Create data frame and calculate relative values.
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

def parse_tracked_params(system_file):

    """Function which parses any additional parameters to be tracked from an input file

        Input:
         system_file: str - name of input .txt file which contains any additional parameters to be tracked across the scan [indexes are expected to be the gaussian indexes]

         Format of input file:
             param_name (atom_types) atom1_ind atom2_ind [atom3_ind atom4_ind]
             E.g. OPSC 3 1 2 7

        Returns:
         tracked_params: dict:
                         key: str - param_name
                         value: list of ints - [atom_indexes]
    """

    # Initialise empty dict for params
    tracked_params = {}
    # Parse in file and seperate the indexes from the parameter ID and save as an entry to the dict
    with open(system_file, 'r') as infile:
        for el in infile:
            param = el.strip().split(' ')
            indexes = [int(ind)-1 for ind in param[1:]]
            tracked_params[param[0]] = indexes
    return tracked_params


def init_scan(*args, tracked_params=None):

    """Function that generates a list of molecule objects from a scan file

    Parameters:
     args: str - gaussian log files of scan results
     tracked_params: [optional] str - file with tracked parameters in (gaussian indexes)

    Returns:
     scan_molecules: List of Molecule objects for each step of scan
    """

    # Inititalise variables
    scan_molecules = []

    # Parse in tracked parameters if set
    if tracked_params != None:
        parameters = parse_tracked_params(tracked_params)
    else:
        parameters = {}

    for i, input_file in enumerate(args):

        # Initialise gaussian log file object
        scan_file = pgauss.GaussianLog(input_file)

        # Get scan info - depending if rigid or relaxed scan
        if scan_file.job_type == 'scan_relaxed':
            scan_info = scan_file.get_scan_info()
            # Set scan parameter in parameters dict and range of opt steps in file
            parameters[scan_info['param_key']] = scan_info['atom_inds']
            opt_steps = list(range(1, scan_info['num_steps']+2))
        else:
            # If rigid scan then needs more processing as potentially has mutltiple scan parameters
            scan_info = scan_file.get_rigid_scan_info()
            total_scan_steps = 1
            # Set scan parameters in parameters dict and range of opt steps in file
            for scan_parameter in list(scan_info.values()):
                parameters[scan_parameter['param_key']] = scan_parameter['atom_inds']
                total_scan_steps *= (scan_parameter['num_steps'] + 1)
            opt_steps = list(range(1, total_scan_steps+1))

        # scan_molecules += molecules.init_mol_from_log(input_file, opt_steps=opt_steps, parameters=parameters)

    return scan_molecules, scan_info

def sum_mols(*args):

    """
    Want to take a list of molecules, set properties as first one,
    and then add the others. 
    Single numeric items can be summed.
    Discrete data will need to be put in to a list?
    
    Whether to create a new molecule object altogether - would need new init
    possibility just to set all values.
    Or make new class which is a collection of molecules?

    """

    # Set sums for quantities and empty lists.
    escf_sum = 0.0
    atom_list, log_files = [], []
    optimised = True
    thermo = False

    # Check if Molecue/MoleculeThermo object for summing thermo properties or not.
    if hasattr(args[0], 'e'):
        thermo = True
        # Thermo sums in order of e, h, g, s, zpe - should proabbly make as dict for consistency with other methods.
        thermo_sums = [0.0, 0.0, 0.0, 0.0, 0.0]

    # Add values for each molecule to quantity sums
    for mol in args:

        # Combine shared Molecue/MoleculeThermo properties of logfile, atom ids and SCF energy.
        log_files.append(mol.file_name)
        atom_list.append(mol.atom_ids)
        escf_sum += mol.escf

        # Check if molecules are optimised
        if mol.optimised == False:
            optimised = False
            print('Warning, one molecule in complex is not optimised')

        # Sum thermodynamic values if present.
        if thermo == True:
            try:
                for i, thermo_val in enumerate([mol.e, mol.h, mol.g, mol.s, mol.zpe]):
                    thermo_sums[i] += thermo_val
            except AttributeError:
                print('Molecule does not have correct thermodynamic values to be summed')

    # Change logfile list.
    log_files = ','.join(map(str, log_files))

    # Instantiate molecule class with summed values - not sure if summing ZPE is physical.
    if thermo == True:
        new_mol = molecules.MoleculeThermo(log_files, mol_energy=escf_sum, mol_geom=None, atom_ids=atom_list, optimised=optimised, e=thermo_sums[0], h=thermo_sums[1], g=thermo_sums[2], s=thermo_sums[3])
    else:
        new_mol = molecules.Molecule(log_files, mol_energy=escf_sum, mol_geom=None, atom_ids=atom_list, optimised=optimised,)

    return new_mol

def process_input_file(input_file):
    """
    Process an input file to Molecule Objects and/or a DataFrame.
    
    If a .conf file is given the Molecule objects are processed for each entry in the .conf file.
    If a .csv file then the information is parsed directly into a moleucle DataFrame.
    
    Parameters
    ----------
    input_file: `str`
        File name with file type of: .conf or .csv

    Returns
    -------
    mol_df: pandas DataFrame`
        Molecule information and properties.
    molecules: `list of Molecule object`
        Molecule objects for each entry line in .conf file.
        [Only returned if input file type is conf.
    
    """
    # Retrieve file type for input file.
    file_type = str(input_file.split('.')[-1])

    # Process conf file, creating Molecule objects and a DataFrame.
    if file_type == 'conf':
        mol_names, molecules = construct_mols(input_file)
        mol_df = mols_to_dataframe(molecules, mol_names=mol_names)
        return mol_df, molecules

    # Parse in existing dataframe and set first column (mol_names) as index.
    elif file_type == 'csv':
        mol_df = pd.read_csv(input_file, index_col=0)
        return mol_df

    # Raise exception if file type is not recognised.
    else:
        raise Exception('File extension not recognised (should be .conf or .csv)')
