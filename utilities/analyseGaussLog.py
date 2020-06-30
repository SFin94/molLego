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

        '''Initialises class, runs methods to set attributes about the calculation (molecule an job details) and flags for parsing properties from the file
        '''

        self.file_name = logfile
        self.set_job_details()
        self.set_flags()


    def set_job_details(self):
        
        '''Class method to assign calculation details from output
        Sets class attributes:
         method: :class:`str` - functional/method of calculation
         basis_set: :class:`str` - basis set input of calculation
         job_type: :class:`str` - calculation type (Opt, Fopt, Freq, SP, Scan)
         sp: :class:`bool` - flag of whether the calculation is an optimisation or single point energy calculation

        Also runs set_molecule which sets the charge, numer of atoms, elements and atom ids 
        '''

        # Parse job and molecule information from log file
        raw_output, modred = self.pull_job_details()

        # Process raw_output and set job attributes
        raw_output = raw_output.split('\\')
        self.job_type = raw_output[3].lower()
        self.method, self.basis_set  = raw_output[4:6]
        self.set_molecule(raw_output[6])

        # Set spe flag if single point and no optimisation steps
        self.spe = (self.job_type.lower() == 'sp')
        
        # Set different job types for rigid and relaxed scans
        if self.job_type.lower() == 'scan':
            if modred == True:
                self.job_type = 'scan_relaxed'
            else:
                self.job_type = 'scan_rigid'
                self.spe = True


    def set_flags(self):

        '''Class method that converts the job type to a set of flags showing what information is contained in the file.
        Sets class attributes:
         job_property_flags: :class:`dict` - key is property type and value is corresponding string flag for parsing property from gaussian log file

        Parameters:
         target_property: list of str - target property/ies wanted from log file; overrides dafault scheme of using job type to set them
        '''

        # Dict mapping job type to the properties contained in the log file
        job_to_property = {'opt': ['energy', 'geom', 'opt'], 'freq': ['energy', 'geom','thermo', 'opt'], 'fopt': ['energy', 'geom', 'thermo', 'opt'], 'sp': ['energy', 'geom'], 'scan_relaxed': ['energy', 'geom', 'opt'], 'scan_rigid': ['energy', 'geom']}

        # Dict of search flags in log file for each property
        property_flags = {'energy': 'SCF Done', 'geom': 'Standard orientation', 'opt': 'Optimized Parameters', 'thermo': 'Thermochemistry'}

        # Set dict of search flags for specific job type depending on properties calculated
        try:
            flag_keys = job_to_property[self.job_type.lower()]
            self.job_property_flags = {prop: property_flags[prop] for prop in flag_keys}
        except:
            print('Job type is not recognised')

        # Switch energy flag if method is MP2
        if 'MP2' in self.method:
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
        self._set_charge(mol_formula)

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
        self.atom_ids = self.pull_atom_ids()


    def pull_atom_ids(self):

        '''Class method to pull atom IDs from start of gaussian log file

        Returns:
         atom_ids: list of str - atom IDs of the molecule
        '''

        # Initialise variables
        atom_ids = []
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
                        atom_ids.append(line.split()[0][0])
                        line = input.__next__()
    
                    return atom_ids


    def _pull_job(self, input, current_line):

        '''Class method to pull final block of information from gaussian log file
        
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
        final_info_flag = '1\\1\\'
        modredundant_flag = 'modredundant'
        modred = False

        # Iterate through file and pulls final section
        with open(self.file_name, 'r') as input:
            for line in input:
                if final_info_flag in line:
                    job_details = self._pull_job(input, line)
                elif modredundant_flag in line.lower():
                    modred = True

        return job_details, modred


    def _pull_geometry(self, input, current_line):

        '''Class method to pull cartesian coordinate geometry from standard orientation section in gaussian log file

        Parameters:
         input: iter object - lines of file
         current_line: current line in file

        Returns:
         atomCoords: numpy array - xyz coordinates for each atom
        '''
        
        # Initalise variables
        atomCoords = []
        # Skip the header section of the standard orientation block
        [input.__next__() for x in range(0,4)]
        # Parse the atomic coordinates
        for atom in range(self.atom_number):
            line = input.__next__()
            xyz = np.asarray([
                float(line.split()[jind+3])
                for jind in range(3)
            ])
            atomCoords.append(xyz)

        return np.asarray(atomCoords)


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
            temp - temperature of 
            totE - thermally corrected energy (kJ/mol)
            totH - enthalpy (kJ/mol)
            totG - free energy (kJ/mol)
            totS - entropy (kJ/mol)
            zpe - zero point energy (kJ/mol)
        '''

        # Skip to temperature line and set temperature
        for i in range(2):
            line = input.__next__()
        temp = float(line[15:22])

        # Skip line until thermochemistry section reached
        while 'Zero-point correction' not in line:
            line = input.__next__()
        thermo = [float(line[50:58])]

        # Skip to thermally corrected e, h, g lines
        [input.__next__() for x in range(0,3)]
        for i in range(3):
            line = input.__next__()
            thermo.append(float(line[53:].strip()))
        thermo.append((thermo[2]-thermo[3])/temp)

        # Convert to kJ/mol
        return [value*2625.5 for value in thermo]


    def _update_opt_count(self, property):

        '''Class method to update the count of the current geometry number in the log file, updates differently whether calcualtion is an optimisation (updates once count for each optimised geometry) or if calculation is a single point (rigid scan) (updates for each new geometry)

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


    def pull_properties(self, opt_steps=[1], target_property=None):
        
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
        if 'MP2' in self.job_type:
            pull_functions['energy'] = self._pull_mp2_energy

        # Open and iterate through log file
        with open(self.file_name, 'r') as input:
            for line in input:
                # Check if any property is in the line and set property value
                for property, flag in self.job_property_flags.items():
                    if flag in line:
                        step_result[property] = pull_functions[property](input, line)
                        opt_count += self._update_opt_count(property)
                
                # If target optimisation step is met append results
                if (opt_count == opt_steps[opt_step_ind]):
                    opt_step_ind += 1
                    mol_results[opt_count] = step_result
                    step_result = {}
                
                    # Return if results calculated for all optimisation steps
                    if opt_step_ind == len(opt_steps):
                        return mol_results


    def _pull_modredundant(self, input):

        '''Class method that pulls the scan information from the log file

        Parameters: 
         input: iter object - lines of file

        Returns: 
         modred_input: list of str - list of all moredundant input lines from log file
        '''

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
            scan_info['param_key'] += (self.atom_ids[int(i)] + i + '-')
        scan_info['param_key'] = scan_info['param_key'][:-1]

        return scan_info


    def set_scan_info(self):

        '''Class method that sets scan information for relaxed scan from a gaussian log file

        Returns:
        scan_info: dict -
            {param_key: str - '-' seperated atom id + index of all atoms in scan parameter e.g. 'H1-O2'
            atom_inds: list of int - indexes of the atoms in scan parameter,
            num_steps: int - number of scan steps
            step_size: float - size of the scan step}
        '''

        # Set modredundant information flag
        modred_flag = 'The following ModRedundant input section has been read:'

        # Iterate over file and pull modredundant section
        with open(self.file_name, 'r') as input:
            for line in input:
                if modred_flag in line:
                    modred_input = self._pull_modredundant(input)

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






