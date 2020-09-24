import sys
import numpy as np
import pandas as pd

'''
Module containing parsing functions to get information from gaussian log files.
'''

class GaussianLog():

    '''Class for a gaussian log file

    Class attributes
        file_name: :class:`str` - filepath/name of parent log file
        method: :class:`str` - functional/method of calculation
        basis_set: :class:`str` - basis set input of calculation
        job_type: :class:`str` - calculation type (Opt, Fopt, Freq, SP, Scan)
        sp: :class:`bool` - flag of whether the calculation is an optimisation or single point energy calculation
        self.charge: :class:`int` - charge of the molecule
        atom_number: :class:`int` - number of atoms in molecule
        elements: :class:`list of str` - list of elements present in molecule
        job_property_flags: :class:`dict` - key is property type and value is corresponding string flag for parsing property from gaussian log file
    '''

    def __init__(self, logfile):

        '''Initialises class, runs methods to set attributes about the calculation (molecule and job details) and flags for parsing properties from the file
        '''

        self.file_name = logfile
        self.set_job_details()
        self.set_flags()


    def _pull_job_start(self, input, current_line):

        '''Class method to parse job details - an alternative to the end method which is preferred (more information available) in the case that the job is not archived and the final block is not printed (E.g. relazed scan with  non-optimized steps)

        Parameters:
         input: iter object - lines of file
         current_line: current line in file
        
        Returns:
         output: str - information from the end of the file containing some job details (method, basis set, job type and molecular formula)
        '''

        # Initialise variables and skip to start line
        output = ''
        while '---' not in current_line:
            current_line = input.__next__()
        current_line = input.__next__()
        
        # Parse job input lines
        while '---' not in current_line:
            output += (current_line.strip())
            current_line = input.__next__()

        return output


    def _pull_job_end(self, input, current_line):

        '''Class method to pull final block of information from gaussian log file

        Parameters:
         input: iter object - lines of file
         current_line: current line in file
        
        Returns:
         output: str - str - information from the end of the file containing some job details (method, basis set, job type and molecular formula)
        '''

        # Initialise variables
        output = ''
        
        # Add relevant lines to output string to collect all information wanted
        while '\\\\' not in current_line:
            output += (current_line.strip())
            current_line = input.__next__()
        output += current_line.split('\\\\')[0]

        return output

    
    def pull_job_details(self):

        '''Class method that pulls job details from the gaussian log file
        Sets private class attribute:
         _modred: :class:`bool` - flag to tell rigid and relaxed scans apart

        Returns:
         job_details: str - information from the end of the file containing some job details (method, basis set, job type and molecular formula)
        '''
        
        # Set flag for the section to parse
        end_info_flag = '1\\1\\'
        pull_info_flags = {'%mem=': self._pull_job_start, end_info_flag: self._pull_job_end}
        extra_info_flags = {'modredundant': False, 'scan ': False}

        # Iterate through file and pull job details
        with open(self.file_name, 'r') as input:
            for line in input:

                # Set flags for modredundant input or scan information
                for flag in list(extra_info_flags.keys()):
                    if flag in line.lower():
                        extra_info_flags[flag] = True

                # Pull start and end information (if present) from file
                for flag in list(pull_info_flags.keys()):
                    if flag in line.lower():
                        job_details = pull_info_flags[flag](input, line)

                        # Return if end of job information - first instance for composite jobs
                        if flag == end_info_flag:
                            return job_details, extra_info_flags

        return job_details, extra_info_flags


    def _process_job_start(self, raw_output, extra_output):

        '''Class method that prcoesses start job output line to set several attributes for the log file. Less information is set and is a back up method for processing the end job information
        
        Parameters:
         raw_output: str - raw string output of input job spec in gaussian log file output
        
        Sets class attributes:
         job_type: :class:`str` - calculation type (Opt, Fopt, Freq, SP, Scan)
         sp: :class:`bool` - flag of whether the calculation is an optimisation or single point energy calculation
        '''

        # Uses calculation flags to try to deduce what job type the calculation is
        calculation_flags = {'opt': False, 'freq': False, 'scan': False, 'mp2': False}
        job_calculation_types = {'opt': [True, False, False], 'fopt': [True, True, False], 'freq': [False, True, False], 'scan_rigid': [False, False, True]}

        # Set calculation flag to True if present in job input
        for flag in list(calculation_flags.keys()):
            if flag in raw_output.lower():
                calculation_flags[flag] = True
        
        # Set job type based on calculation bool results
        for job, calc_type in job_calculation_types.items():
            if calc_type == list(calculation_flags.values())[:-1]:
                self.job_type = job
                break

        # If both scan and modredundant flags are true then job is a relaxed scan
        if all(list(extra_output.values())):
            self.job_type = 'scan_relaxed'

        # Set MP2 as method if present for correct energy parsing
        if calculation_flags['mp2'] == True:
            self.method = 'mp2'

        # Set number of atoms and atom ids
        self.pull_atom_number()
        self.pull_atom_ids()


    def _process_job_end(self, raw_output, extra_output):

        '''Class method that prcoesses end job output line to set several attributes for the log file
        
        Parameters:
         raw_output: str - raw string output of final gaussian log file output containing job information
        
        Sets class attributes:
         method: :class:`str` - functional/method of calculation
         basis_set: :class:`str` - basis set input of calculation
         job_type: :class:`str` - calculation type (Opt, Fopt, Freq, SP, Scan)
         sp: :class:`bool` - flag of whether the calculation is an optimisation or single point energy calculation
        '''

        raw_output = raw_output.split('\\')
        self.job_type = raw_output[3].lower()
        self.method, self.basis_set  = raw_output[4:6]
        self.set_molecule(raw_output[6])
        
        # Set different job types for rigid and relaxed scans
        if self.job_type.lower() == 'scan':
            if extra_output['modredundant'] == True:
                self.job_type = 'scan_relaxed'
            else:
                self.job_type = 'scan_rigid'


    def set_job_details(self):
        
        '''Class method to assign calculation details from output
        Sets class attributes:
         method: :class:`str` - functional/method of calculation
         basis_set: :class:`str` - basis set input of calculation
         job_type: :class:`str` - calculation type (Opt, Fopt, Freq, SP, Scan)
         spe: :class:`bool` - flag of whether the calculation is an optimisation or single point energy calculation

        Also runs set_molecule which sets the charge, numer of atoms, elements and atom ids 
        '''

        # Parse job and molecule information from log file
        raw_output, extra_output = self.pull_job_details()
        
        # Test if raw output is from end information from successful job for processing
        if '1\\1\\' in raw_output:
            self._process_job_end(raw_output, extra_output)

        # Otherwise try processing less information from job input at the start of the file
        else:
            try:
                self._process_job_start(raw_output, extra_output)
                print('Normal termination output not present, fewer attributes set on input information')
            except:
                print('Cannot parse job information from log file')
        
        # Set spe flag if single point and no optimisation steps
        self.spe = any([(self.job_type.lower() == 'sp'), (self.job_type.lower() == 'scan_rigid')])


    def set_flags(self):

        '''Class method that converts the job type to a set of flags showing what information is contained in the file.
        Sets class attributes:
         job_property_flags: :class:`dict` - key is property type and value is corresponding string flag for parsing property from gaussian log file

        Parameters:
         target_property: list of str - target property/ies wanted from log file; overrides dafault scheme of using job type to set them
        '''

        # Dict mapping job type to the properties contained in the log file
        job_to_property = {'opt': ['energy', 'geom', 'opt'], 'freq': ['energy', 'geom', 'thermo', 'opt'], 'fopt': ['energy', 'geom', 'thermo', 'opt'], 'sp': ['energy', 'geom'], 'scan_relaxed': ['energy', 'geom', 'opt'], 'scan_rigid': ['energy', 'geom']}

        # Dict of search flags in log file for each property
        property_flags = {'energy': 'SCF Done', 'geom': 'Standard orientation', 'opt': 'Optimized Parameters', 'thermo': 'Thermochemistry'}

        # Set dict of search flags for specific job type depending on properties calculated
        try:
            flag_keys = job_to_property[self.job_type.lower()]
            self.job_property_flags = {prop: property_flags[prop] for prop in flag_keys}
        except:
            print('Job type is not recognised')

        # Switch energy flag if method is MP2 - extra handle in case method was not parsed
        if (hasattr(self, 'method')):
            if ('mp2' in self.method.lower()):
                self.job_property_flags['energy'] = 'EUMP2'


    def _set_charge(self, mol_formula):

        '''Class method used by set_molecule to set the molecules charge
        Sets class attributes:
         self.charge: :class:`int` - charge of the molecule
        '''

        # Set charge as neutral if none given
        if mol_formula.isalnum():
            self.charge = 0
        # Set charge if given and remove from molecular formula
        else:
            self.charge = int(mol_formula.split('(')[1][:-2])
            if mol_formula[-2] =='-':
                self.charge *= -1
        mol_formula = mol_formula.split('(')[0]
        
        return mol_formula


    def _set_character(self, char):

        '''Class method used by set_molecule to sort the elements from the number of atoms
        Updates class attributes:
         atom_number: :class:`int` - number of atoms in molecule
         elements: :class:`list of str` - list of elements present in molecule
        '''

        # Check type of character from molecular formula and set to attribute
        if char.isdigit():
            self.atom_number += int(char)
        else:
            self.elements.append(char)


    def set_molecule(self, mol_formula):
        
        '''Class method to count number of atoms in system from a molecular formula
        Sets class attributes:
         atom_number: :class:`int` - number of atoms in molecule
         elements: :class:`list of str` - list of elements present in molecule'''

        # Initialise variables
        self.atom_number = 0
        self.elements = []
        i = 0

        # Set charge and remove any charge info from molecular formula
        mol_formula = self._set_charge(mol_formula)

        while i < len(mol_formula)-1:
            char = mol_formula[i]
            while char.isdigit() == mol_formula[i+1].isdigit():
                char += (mol_formula[i+1])
                i += 1
            self._set_character(char)
            i += 1
        # Add last entry if it wasn't part of another character
        if i != len(mol_formula):
            self._set_character(mol_formula[-1])

        # Set atom ids - relies on self.atom_number already being set
        self.pull_atom_ids()


    def pull_atom_ids(self):

        '''Class method to pull atom IDs from start of gaussian log file

        Sets class attributes:
         atom_ids: :class:`list of str` - atom IDs of the molecule
        '''

        # Initialise variables
        self.atom_ids = []
        atom_id_flag = 'Charge = '
        jump_line_flags = ['No Z-Matrix', 'Redundant internal coordinates']

        # Iterate through file and pulls final section
        with open(self.file_name, 'r') as input:
            for line in input:
                if atom_id_flag in line:

                    # Get to start of input section and pull the atom ids from the list
                    line = input.__next__()
                    if any(flag in line for flag in jump_line_flags):
                        line = input.__next__()
                    for atom in range(self.atom_number):
                        self.atom_ids.append(line.split()[0][0])
                        line = input.__next__()
                    break


    def _pull_geometry(self, input, current_line):

        '''Class method to pull cartesian coordinate geometry from standard orientation section in gaussian log file

        Parameters:
         input: iter object - lines of file
         current_line: current line in file

        Returns:
         atom_coords: numpy array - xyz coordinates for each atom
        '''
        
        # Initalise variables
        atom_coords = []
        # Skip the header section of the standard orientation block
        [input.__next__() for x in range(0,4)]
        # Parse the atomic coordinates
        for atom in range(self.atom_number):
            line = input.__next__()
            xyz = np.asarray([
                float(line.split()[i+3])
                for i in range(3)
            ])
            atom_coords.append(xyz)

        return np.asarray(atom_coords)


    def _pull_energy(self, input, current_line):

        '''Class method to pull energy from gaussian log file

        Parameters:
         input: iter object - lines of file
         current_line: current line in file
         
        Returns:
         energy: float - SCF Done energy (a.u.)
        '''

        # Parse energy from input line
        energy = float(current_line.split('=')[1].split()[0])

        return energy


    def _pull_mp2_energy(self, input, current_line):
    
        '''Class method to pull energy from gaussian log file

        Parameters:
         input: iter object - lines of file
         current_line: current line in file
         
        Returns:
         mp2_energy: float - SCF Done energy (a.u.)
        '''

        mp2_raw = current_line.split('=')[2].strip()
        mp2_energy = float(mp2_raw.split('D')[0])*np.power(10, float(mp2_raw.split('D')[1]))

        return mp2_energy


    def _pull_optimised(self, input, current_line):

        '''Class method to pull optimised information from gaussian log file

        Parameters:
         input: iter object - lines of file
         current_line: current line in file
         
        Returns:
         opt: bool - whether the step is optimised or not
        '''

        opt = ('Non-Optimized' not in current_line)

        return opt


    def _pull_thermo(self, input, current_line):
        
        '''Class method to pull energy from gaussian log file

        Parameters:
         input: iter object - lines of file
         current_line: current line in file
         
        Returns:
         list of floats:
            temp - temperature 
            zpe - zero point energy (kJ/mol)
            totE - thermally corrected energy (kJ/mol)
            totH - thermally corrected enthalpy (kJ/mol)
            totG - thermally corrected free energy (kJ/mol)
            totS - entropy (kJ/mol)
        '''

        # Initialise variables
        quantities = ['T', 'ZPE', 'E', 'H', 'G', 'S']
        thermochemistry = {quantity: 0.0 for quantity in quantities}

        # Skip to temperature line and set temperature
        for i in range(2):
            line = input.__next__()
        thermochemistry['T'] = float(line[15:22])

        # Skip to ZPE line and set ZPE
        while 'Zero-point correction' not in line:
            line = input.__next__()
        thermochemistry['ZPE'] = float(line[50:58])

        # Optional section if thermal corrections to E, H, G wanted
        # thermal_corrections = [float(input.__next__()[50:58]) for i in range(3)]
        # return thermal_corrections
        
        # Skip to thermally corrected values and set thermally corrected E, H, G
        [input.__next__() for x in range(0,4)]
        for quantity in quantities[2:-1]:
            thermochemistry[quantity] = float(input.__next__()[53:].strip())

        # Calculate TdS
        thermochemistry['S'] = (thermochemistry['H'] - thermochemistry['G'])/thermochemistry['T']

        # Convert to kJ/mol
        for quantity in thermochemistry:
            thermochemistry[quantity] *= 2625.5
            
        return thermochemistry


    def _update_opt_count(self, property):

        '''Class method to update the count of the current geometry number in the log file, updates differently whether calculation is an optimisation (updates once count for each optimised geometry) or if calculation is a single point (rigid scan) (updates for each new geometry)

        Parameters:
         property: str - the current property being parsed from the log file

        Returns:
         bool - (acts as 0/1 value for updating the opt count)
        '''
        
        if property == 'energy':
            return self.spe
        elif property == 'opt':
            return (not self.spe)
        else:
            return False


    def pull_properties(self, opt_steps=None, target_property=None, scale_factor=1.0):
        
        '''Class method to parse the energy, thermodynamic data, geometry and optimised information from specified optimsation step/s in the log file for a molecule.

        Parameters:
         opt_steps: list of int - target optimisation/geometry step/s wanted from the scan (rigid scan sp=True; relaxed scan sp=False) or optimisation trajectory (sp=True)

        Returns:
         mol_results: dict of dicts-  containing target properties for the molecule with the key as the optimisation/sp count [opt_steps]
        '''

        # Initialise variables
        step_result = {}
        opt_count = 0
        opt_step_ind = 0
        mol_results = {}
        
        # Mapping of functions to property
        pull_functions = {'energy': self._pull_energy, 'geom': self._pull_geometry, 'thermo': self._pull_thermo, 'opt': self._pull_optimised}
        if self.job_property_flags['energy'] == 'EUMP2':
            pull_functions['energy'] = self._pull_mp2_energy

        # Set opt count to 2 if fopt calculation as thermo occurs after opt count met
        if opt_steps == None:
            job_types = ('freq', 'opt', 'fopt')
            try:
                opt_steps = [job_types.index(self.job_type)]
            except:
                opt_steps = [1]

        # Open and iterate through log file
        with open(self.file_name, 'r') as input:
            for line in input:
                # Check if any property is in the line and set property value
                for property, flag in self.job_property_flags.items():
                    if flag in line:
                        step_result[property] = pull_functions[property](input, line)
                        opt_count += self._update_opt_count(property)
                
                # If target optimisation step is met append results
                if (opt_count == opt_steps[opt_step_ind]) and opt_steps[opt_step_ind] >= 1:
                    opt_step_ind += 1
                    mol_results[opt_count] = step_result
                    step_result = {}
                
                    # Return if results calculated for all optimisation steps
                    if opt_step_ind == len(opt_steps): 
                        return mol_results
            
            # Append final step results (needed for frequency calculation)
            mol_results[opt_count] = step_result
            return mol_results


    def _pull_modredundant(self, input):
        """
        Pull scan information from the log file.

        Parameters: 
         input: iter object - lines of file

        Returns: 
         modred_input: list of str - list of all moredundant input lines from log file
        """
        # Goes to start of modredundant section then extracts all modredundant lines
        modred_input = []
        current_line = input.__next__()
        while current_line.strip() != '':
            modred_input.append(current_line.strip().split())
            current_line = input.__next__()
        return modred_input
    

    def _process_modredundant(self, modred_input):

        '''Class method to process the scan information from the modredundant input

        Parameters:
        modred_input: nested list of str - unprocessed modredundant input lines from log file

        Returns:
         scanInfo: dict -
            {param_key: str - '-' seperated atom id + index of all atoms in scan parameter e.g. 'H1-O2'
            atom_inds: list of int - indexes of the atoms in scan parameter,
            num_steps: int - number of scan steps
            step_size: float - size of the scan step}
        '''
        
        # Initialise variables
        scan_info = {'param_key':'', 'atom_inds':[]}
    
        # Locates scan input line from modredundant input by finding 'S' character
        for modred_line in modred_input:
            if 'S' in modred_line:
                scan_input = modred_line

        # Process scan information from modredundant input
        scan_info['num_steps'] = int(scan_input[-2])
        scan_info['step_size'] = float(scan_input[-1])

        # Set atom indexes (-1 for python index) and parameter key ('-' seperated, atom id, atom index)
        s_index = scan_input.index('S')
        for i in scan_input[1:s_index]:
            scan_info['atom_inds'].append(int(i) - 1)
            scan_info['param_key'] += (self.atom_ids[int(i)-1] + i + '-')
        scan_info['param_key'] = scan_info['param_key'][:-1]

        return scan_info


    def set_scan_info(self):
        """
        Set scan information for relaxed scan from a gaussian log file.

        Returns:
        scan_info: dict -
            {param_key: str - '-' seperated atom id + index of all atoms in scan parameter e.g. 'H1-O2'
            atom_inds: list of int - indexes of the atoms in scan parameter,
            num_steps: int - number of scan steps
            step_size: float - size of the scan step}
        """
        # Set modredundant information flag
        modred_flag = 'The following ModRedundant input section has been read:'

        # Iterate over file and pull modredundant section
        with open(self.file_name, 'r') as input:
            for line in input:
                if modred_flag in line:
                    modred_input = self._pull_modredundant(input)
                    break

        # Process scan information from logfile modredundant input
        scan_info = self._process_modredundant(modred_input)

        return scan_info


    def _pull_rigid_scan(self, input):

        '''Class method that pulls the rigid scan information from the log file

        Parameters: 
         input: iter object - lines of file

        Returns: 
         initial_zmat: list of str - the initial z matrix for the molecule
         variables: list of str - unprocessed lines of information for any set variables in the z matrix
        '''

        # Initialise variables
        initial_zmat = []
        variables = []
        variables_flag = 'Variables'

        # File is ordered to give initial z matrix, a header 'Variables' and then the variables with no blank lines in between
        current_line = input.__next__()

        # Pull all lines of the initial z matrix from the file
        while variables_flag not in current_line:
            initial_zmat.append(current_line.strip())
            current_line = input.__next__()
        current_line = input.__next__()

        # Pull all variables lines from file
        while current_line.strip() != '':
            variables.append(current_line.split())
            current_line = input.__next__()
        
        return variables, initial_zmat


    def _process_rigid_scan(self, variables, initial_zmat):

        '''Class method that processes raw rigid scan input to scan information for each scan parameter

        Parameters:
            initial_zmat: list of str - the initial z matrix for the molecule
            variables: list of str - unprocessed lines of information for any set variables in the z matrix
            
        Returns:
        scanInfo: dict of dicts -
            Has an entry for each scan parameter where key is the variable name in the zmatrix and value:
            {param_key: str - '-' seperated atom id + index of all atoms in scan parameter e.g. 'H1-O2'
            atom_inds: list of int - indexes of the atoms in scan parameter,
            num_steps: int - number of scan steps
            step_size: float - size of the scan step}
        '''

        # Intialise variables
        total_scan_steps = 1
        scan_info = {}

        # Iterate over the variable entries to locate scan parameters
        for var_line in variables:
            if any('Scan' in var for var in var_line):

                # Set the number of steps and step size for each scanned parameter
                num_steps = int(var_line[2])
                total_scan_steps *= (num_steps+1)
                step_size = float(var_line[3])
                scan_info[var_line[0]] = {'num_steps': num_steps, 'step_size': step_size}
        
        # Find zmatrix line containing each scan variable
        for atom, atom_zmat in enumerate(initial_zmat):
            for scan_variable in list(scan_info.keys()):
                if scan_variable in atom_zmat:

                    # Set scan parameter and atom inds (python index) from initial z matrix
                    atom_inds = [atom]
                    param_key = self.atom_ids[atom] + str(atom+1)
                    atom_entry = atom_zmat.split()
                    for i in range(1, atom_entry.index(scan_variable), 2):
                        index = int(atom_entry[i]) - 1
                        atom_inds.append(index)
                        param_key += ('-' + self.atom_ids[index] + atom_entry[i])

                    # Set the scan parameter key and atom inds for each variable in the scan_info dict
                    scan_info[scan_variable]['atom_inds'] = atom_inds
                    scan_info[scan_variable]['param_key'] = param_key
                    
        return scan_info


    def set_rigid_scan_info(self):

        '''Class method that sets scan information for rigid scan from a gaussian log file

            Parameters:

            Returns:
            scanInfo: dict -
                {param_key: str - '-' seperated atom id + index of all atoms in scan parameter e.g. 'H1-O2'
                atom_inds: list of int - indexes of the atoms in scan parameter,
                num_steps: int - number of scan steps
                step_size: float - size of the scan step}
        '''
        
        zmat_start_flag = 'Charge'

        # Iterate through file and pull relevant input sections
        with open(self.file_name, 'r') as input:
            for line in input:
                if zmat_start_flag in line:
                    variables, initial_zmat = self._pull_rigid_scan(input)
                    break
        rigid_scan_info = self._process_rigid_scan(variables, initial_zmat)

        return rigid_scan_info


    def pull_atom_number(self):

        '''Class method to pull the number of atoms - back up method as should be set when setting job details for a successful job

        Class attributes set:
         atom_number: :class:`int` - number of atoms in molecule
        '''

        with open(self.file_name, 'r') as input:
            for line in input:
                if 'natoms' in line.lower():
                    self.atom_number = int(line.split()[1])
                    break


    def pull_multiplicity(self):

        '''Class method to pull charge and multiplicity

        Returns:
         multiplicity: :class:`int` - multipliicity of molecule
        '''

        with open(self.file_name, 'r') as input:
            for line in input:
                if 'multiplicity' in line.lower():
                    return int(line.split()[-1])


