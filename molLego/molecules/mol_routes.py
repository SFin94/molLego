"""Module of routines that use Molecule/MoleculeThermo/ReactionPath objects."""
import sys
import numpy as np
import pandas as pd

import molLego.parsers.parse_gaussian as pgauss
import molLego.utilities.geom as geom
import molLego.molecules.molecules as molecules

def construct_mols(system_file):
    """
    Create Molecule or MoleculeThermo object for each molecule in a system conf file.

    Parameters:
     system_file: str - name of the system file containing the molecule names/keys and the log files to be parsed
            The file should be formatted:
                mol_key mol_output.log
        Lines can be commented out with a #

    Returns:
     mol_names: list of str - molecule names/keys for each molecule in file [mol_key in system file]
     mols: list of :Molecule:/:MoleculeThermo: objects for each molecule in system file
    
    """
    # Read in system conf file
    with open(system_file) as input_file:
        input = input_file.read().splitlines()

    # Initialise variables
    mols, mol_names, mol_files = [], [], []

    # Processes contents of .conf file to a list of mol_names and corresponding mol_files
    for line in input:
        if line[0] != '#':
            mol_names.append(line.split()[0])
            mol_files.append(line.split()[1].split(','))

            # Create Molecule or MoleculeThermo object from the input file(s) in each entry of the .conf file
            mols.append(molecules.init_mol_from_log(mol_files[-1][0]))
            for extra_file in mol_files[-1][1:]:
                extra_mol = molecules.init_mol_from_log(extra_file)
                mols[-1] = sum_mols(mols[-1], extra_mol)

    return mol_names, mols

def mols_to_dataframe(mols, mol_names=None, save=None, min=None):

    """Function which creates a dataframe for all of the molecules and can write to a csv

    Parameters:
     mols: list of Molecule or MoleculeThermo objects - instances for each molecule
     mol_names [optional, default=None]: list - molecule names/keys
     save [optional, default=None]: str - name of file to write dataframe to (without csv extension)
     min [optional, default=None]: str - name of the molecule to calculate all values relative too. If not defined then relative values are calculated w.r.t. lowest value for each quantity.

    Returns:
     molecule_df: pandas dataframe - dataframe of all molecules with realtive quantities calcualted
    """

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
    molecule_df = calc_relative(molecule_df, quantities=quantity, min=min)

    # Writes dataframe to file if filename provided
    if save != None:
        molecule_df.to_csv(save + '.csv')
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

        scan_molecules += molecules.init_mol_from_log(input_file, opt_steps=opt_steps, parameters=parameters)

    return scan_molecules, scan_info


def calc_relative(mols_data_full, mols_to_plot=None, quantities=None, min=None):

    """Function to process a dataframe of molecules to plot and calculates relative E SCF (kJ/mol) or Relative E/G/H if thermodynamic properties given

        Parameters:
         mols_data_full: pandas DataFrame - full dataframe for molecules
         mols_to_plot: List of str [optional, default=None] - names (.log file) of conformers to plot from the dataFile
         quantities: list of str [optional, default=None] - The quantitity/ies to plot (str should match dataframe heading). If None, sets to either E, H, G if thermodynamic data or E SCF if not
         min: str [optional, default=None] - index of moelcule to be treated as zero reference

        Returns:
         mols_data: pandas DataFrame - dataframe of the molecules to plot with relative (E SCF)/(E/G/H) columns for plotting
    """

    # Subset amount of data frame to plot
    if mols_to_plot != None:
        mols_data = mols_data_full.reindex(mols_to_plot)
    else:
        mols_data = mols_data_full

    # Calculate relative and normalised quantities
    if quantities == None:
        if 'G' in list(mols_data.columns):
            quantities = ['E', 'H', 'G']
        else:
            quantities = ['E SCF']
    for q in quantities:
        if min != None:
            zero = mols_data[q][min]
        else:
            zero = mols_data[q].min()
        mols_data['Relative '+q] = mols_data[q] - zero

    return mols_data


def sum_mols(*args):

    """Function that adds two molecules together to creat a new one, e.g. for a reactant or product set

    Parameters:
     args: Molecule objects - the molecules to be added

    Returns:
     new_mol - ::class:: object for a molecule

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
