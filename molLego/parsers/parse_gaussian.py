"""Module containing parsing functions for G16 log files."""

import numpy as np
from molLego.parsers.parser import OutputParser
from molLego.utilities.utils import readlines_reverse, parse_mol_formula

# Dict mapping job type to the properties contained in the log file
__job_to_property__ = {
    'opt': ['charge', 'energy', 'geom', 'opt'],
    'freq': ['charge', 'energy', 'geom', 'thermo', 'opt'],
    'fopt': ['charge', 'energy', 'geom', 'thermo', 'opt'],
    'sp': ['charge', 'energy', 'geom'],
    'scan_relaxed': ['charge', 'energy', 'geom', 'opt'],
    'scan_rigid': ['charge', 'energy', 'geom']
    }

# Dict of search flags in log file for each property
__property_flags__ = {
    'charge': 'Charge=',
    'energy': 'SCF Done',
    'geom': 'Standard orientation',
    'opt': 'Optimized Parameters',
    'thermo': 'Thermochemistry',
    }

class LogFileError(Exception):
    """Raised when error in reading log file."""

class GaussianLog(OutputParser):
    """
    Represents a Gaussian log file.

    Attributes
    ----------
    atoms : :class:`list of str`
        The atomic symbols of the atoms in the molecule.

    atom_number : :class:`int`
        The number of atoms in the molecule.

    basis_set : :class:`str`
        The basis set of the calculation.
 
    charge : :class:`int`
        The charge of the molecule.

    elements : :class:`list of str`
        The list of elements present in the molecule.

    file_name : :class:`str`
        The path to the parent log file.

    job_type : :class:`str`
        The calculation type (opt, fopt, freq, sp, scan).

    method : :class:`str`
        The method of the calculation (functional, etc.).

    normal_termination : :class:`Bool`
        ``True`` if normal termination. Otherwise, ``False``.

    """
    
    def __init__(self, output_file):
        """
        Initialise from Gaussian output log file.

        Parameters
        ----------
        output_file : :class:`str`
            The name/path to the parent log file.

        """
        self.file_name = output_file
        self.normal_termination = self._check_normal_termination()

        # Use end of file log for normal termination:
        if self.normal_termination:
            output, job_input = self._pull_end_output()
            output = output.split('\\')

            # Set calculation details.
            self.method, self.basis_set = output[4:6]

            # Set molecule details.
            atom_number, elements, charge = parse_mol_formula(output[6])
            self.atom_number = atom_number
            self.charge = charge
            self.atom_ids = self._pull_atom_ids()

            print(
                'Normal termination output found, '
                'all attributes set.'
                )

        # Try to use the beginning of the file job input data:
        else:
            try:
                job_input = self._pull_start_output()
            
                # Set MP2 as method if present for correct energy parsing.
                if 'mp2' in job_input:
                    self.method = 'mp2'

                # Set number of atoms.
                self.atom_number = self.pull_atom_number()
                
                print(
                    'Normal termination output not present, '
                    'fewer attributes set using input information.'
                    )
            except:
                raise LogFileError(
                    'Cannot parse job information from the log file.'
                    )

        # Set attributes using job input (independant of normal termination).
        self.job_type = self._job_from_input(job_input)

        # Process moderedundant input is present in job input.
        if 'modredundant' in job_input:
            scan_input = self.pull_scan_input()
            if scan_input:
                self.job_type = 'scan_relaxed'

        # Set atom IDs.
        self.atom_ids = self._pull_atom_ids()

    def _check_normal_termination(self):
        """
        Check if the job completed with normal termination.

        Returns
        -------
        :class:`Bool`
            ``True`` if normal termination. Otherwise, ``False``.

        """
        end_line = readlines_reverse(self.file_name)
        next(end_line)
        return True if "Normal termination" in next(end_line) else False

    def _pull_end_output(self):
        """
        Pull start sections of the end job output from normally terminated log file.

        Returns
        -------
        output : `list of str`
            Unprocessed lines from end job output.
            First entry is calculation information.
            Second entry is the job input line.

        """
        # Initialise variables.
        output = ''
        section_count = 0

        with open(self.file_name, 'r') as infile:
            
            for line in infile:
                if '1\\1\\' in line:

                    # Pull the first two sections of end log ouput:
                    while section_count < 2:
                        section_count += ('\\\\' in line)
                        output += line.strip().lower()
                        line = next(infile)
                    break
            return output.split('\\\\')[:2]

    def _pull_start_output(self):
        """
        Pull start job output containing the calculation input from log file.

        Returns
        -------
        output : `str`
            Calculation input line.

        """
        output = ''
        with open(self.file_name, 'r') as infile:
            line = next(infile)
            # Skip to the start log:
            while '#' not in line:
                line = next(infile)
            while '---' not in line:
                output += line.strip().lower()
                line = next(infile)
        return output

    def _job_from_input(self, job_input):
        """
        Set job type from job input.

        Parameters
        ----------
        job_input : `str`
            Job input line for calculation.
        
        Returns
        -------
        :class:`str`
            The calculation type (opt, fopt, freq, sp, scan).
            
        """
        # Use flags to try to deduce the job type.
        calculation_flags = {
            'opt': False,
            'freq': False,
            'scan': False,
            }
        job_calculation_types = {
            'opt': [True, False, False],
            'fopt': [True, True, False],
            'freq': [False, True, False],
            'scan_rigid': [False, False, True],
            'sp': [False, False, False]
            }

        # Set calculation flag to True if present in job input.
        for flag in calculation_flags.keys():
            if flag in job_input.lower():
                calculation_flags[flag] = True

        # Set job type based on calculation bool results.
        for job, calc_type in job_calculation_types.items():
            if calc_type == list(calculation_flags.values()):
                return job

    def _get_search_flags(self):
        """
        Convert the job type to flags for seeking information in the log file.

        Returns
        -------
        job_property_flags : `dict`
            A `dict`, where the key is a property type and the value
            is the corresponding string flag for parsing the property from
            the Gaussian log file.

        """
        # Set parser search flags from the job type.
        try:
            flag_keys = __job_to_property__[self.job_type.lower()]
            job_property_flags = {
                prop: __property_flags__[prop] for prop in flag_keys
                }
        except KeyError:
            print('Job type is not recognised')

        # Switch energy flag if method is MP2.
        if (hasattr(self, 'method')):
            if ('mp2' in self.method.lower()):
                job_property_flags['energy'] = 'EUMP2'

        return job_property_flags

    def _pull_atom_ids(self):
        """
        Pull the atom IDs from start of the log file.

        Returns
        -------
        :class:`list of str`
            The IDs of the atoms in the molecule.

        """
        # Initialise variables.
        atom_ids = []
        atom_id_flag = 'Charge = '
        jump_line_flags = [
            'No Z-Matrix',
            'Redundant internal coordinates'
            ]

        # Iterate through file to input geometry.
        with open(self.file_name, 'r') as infile:
            for line in infile:
                if atom_id_flag in line:
                    line = next(infile)
                    if any(flag in line for flag in jump_line_flags):
                        line = next(infile)
                    # Pull atom IDs.
                    for _ in range(self.atom_number):
                        atom_ids.append(line.split()[0][0])
                        line = next(infile)
                    break

        return atom_ids

    def _pull_charge(self, infile, current_line):
        """
        Pull the charge from the logfile.

        Parameters
        ----------
        infile : iter object
            Lines of file

        current_line
            Current line in file

        Returns
        -------
        :class:`int`
            Formal charge of molecule.

        """
        return int(float(current_line.split()[1]))

    def _pull_geometry(self, infile, current_line):
        """
        Pull the cartesian coordinate geometry from the log file.

        Parameters
        ----------
        infile : iter object
            Lines of file

        current_line
            Current line in file

        Returns
        -------
        :class:`numpy.ndarray`
            A ``(N, 3)`` array of x, y, z positions for each atom.
            Where N is the number of atoms in the molecule.

        """
        # Initalise variables
        atom_coords = []

        # Skip the header section of the standard orientation block
        [next(infile) for x in range(0, 4)]

        # Parse the atomic coordinates
        for _ in range(self.atom_number):
            line = next(infile)
            xyz = np.asarray([
                float(line.split()[i+3])
                for i in range(3)
            ])
            atom_coords.append(xyz)

        return np.asarray(atom_coords)

    def _pull_energy(self, input, current_line):
        """
        Pull the energy from the log file.

        Parameters
        ----------
        infile : iter object
            Lines of file

        current_line
            Current line in file

        Returns
        -------
        :class:`float`
            SCF Done energy (a.u.).

        """
        return float(current_line.split('=')[1].split()[0])

    def _pull_mp2_energy(self, input, current_line):
        """
        Pull the energy from the log file.

        Parameters
        ----------
        infile : iter object
            Lines of file

        current_line
            Current line in file

        Returns
        -------
        :class:`float`
            SCF Done energy (a.u.).

        """
        raw_base, raw_power = current_line.split('=')[2].strip().split('D')
        return float(raw_base)*np.power(10, float(raw_power))

    def _pull_optimised(self, input, current_line):
        """
        Pull optimised information from the log file.

        Parameters
        ----------
        infile : iter object
            Lines of file

        current_line
            Current line in file

        Returns
        -------
        :class:`Bool`
            ``False`` if the step is non-optimised. Otherwise, ``True``.

        """
        return ('Non-Optimized' not in current_line)

    def _pull_thermo(self, input, current_line):
        """
        Pull the thermodynamic information from the log file.

        Parameters
        ----------
        infile : iter object
            Lines of file

        current_line
            Current line in file

        Returns
        -------
        :class:`dict of float`
            The thermodynamic data in the format:
            {
                T   : temperature,
                ZPE : zero point energy (kJ/mol),
                E   : thermally corrected energy (kJ/mol),
                H   : thermally corrected enthalpy (kJ/mol),
                G   : thermally corrected free energy (kJ/mol),
                S   : entropy (kJ/mol)
            }

        """
        # Initialise variables
        quantities = ['T', 'ZPE', 'E', 'H', 'G', 'S']
        thermochemistry = {quantity: 0.0 for quantity in quantities}

        # Skip to temperature line and set temperature
        for _ in range(2):
            line = next(input)
        thermochemistry['T'] = float(line[15:22])

        # Skip to ZPE line and set ZPE
        while 'Zero-point correction' not in line:
            line = next(input)
        thermochemistry['ZPE'] = float(line[50:58])

        # Set the thermally corrected E, H, G.
        [next(input) for x in range(0, 4)]
        for quantity in quantities[2:-1]:
            thermochemistry[quantity] = float(next(input)[53:].strip())

        # Calculate TdS
        thermochemistry['S'] = (thermochemistry['H'] - thermochemistry['G']) \
            / thermochemistry['T']

        # Convert to kJ/mol
        for quantity in thermochemistry:
            thermochemistry[quantity] *= 2625.5

        return thermochemistry

    def get_properties(self):
        """
        Get properties from the log file.

        Parse properties such as the energy, thermodynamic data, 
        geometry, and optimised information from the log file 
        for the molecule.

        Returns
        -------
        :class:`dict`
            A dictionary containing target properties for the molecule.

        """
        # Initialise variables.
        step_result = {}

        # Mapping of functions to property.
        pull_functions = {
            'energy': self._pull_energy,
            'geom': self._pull_geometry,
            'thermo': self._pull_thermo,
            'charge': self._pull_charge,
            'opt': self._pull_optimised
            }

        job_property_flags = self._get_search_flags()

        # Set alternate flag if MP2 energy.
        if job_property_flags['energy'] == 'EUMP2':
            pull_functions['energy'] = self._pull_mp2_energy

        # Open and iterate through log file.
        with open(self.file_name, 'r') as infile:
            for line in infile:
                # Check for property in the line and set property value.
                for prop, flag in job_property_flags.items():
                    if flag in line:
                        step_result[prop] = pull_functions[prop](infile, line)
            
            return step_result

    def _pull_relaxed_scan(self, current_line, infile):
        """
        Pull relaxed scan information from a log file.
        
        Parameters
        ----------
        infile : iter object
            Lines of file

        current_line
            Current line in file

        Returns
        -------
        :class:`list`
            ModRedundant input lines containing scanned parameters.

        """
        # Set modredundant input flag.
        scan_input = []

        current_line = next(infile)
        while current_line.strip():
            if 'S' in current_line:
                scan_input.append(current_line)
            current_line = next(infile)
                    
        return scan_input

    def _process_relaxed_scan(self, scan_input):
        """
        Process scan information from modredundant input.

        Parameters
        ----------
        scan_input : :class:`list` of :class:`str`
            Unprocessed modredundant input scan lines from log file.

        Returns
        -------
        scan : :class:`dict` of :class:`dict`
            An entry for each scan parameter that the PES is a 
            function of. Where Key is the scan variable name and
            Value is a dictionary of scan information of form:
            {
                param_key : :class:`str`
                    '-' seperated atom id + index of all atoms in scan
                    parameter e.g. 'H1-O2'.
                atom_inds : :class:`list of int`
                    indexes of the atoms in scan parameter
                num_steps : :class:`int` - number of scan steps
                step_size : :class:`float` - size of the scan step
            }

        """
        scan_info = {}
        
        # Process scan information from modredundant input.
        for i, line in enumerate(scan_input):

            # Initialise scan step result.
            scan_var = line[0] + str(i)
            scan_info[scan_var] = {
                'param_key': '',
                'atom_inds': []
                }

            line = line.split()
            scan_info[scan_var]['num_steps'] = int(line[-2])
            scan_info[scan_var]['step_size'] = float(line[-1])

            # Set atom indexes (-1 for python index) and parameter key
            # ('-' seperated, atom id, atom index)
            s_index = line.index('S')
            for i in line[1:s_index]:
                scan_info[scan_var]['atom_inds'].append(int(i) - 1)
                scan_info[scan_var]['param_key'] += (
                    f'{self.atom_ids[int(i)-1]}{i}-')
            scan_info[scan_var]['param_key'] = scan_info[
                scan_var]['param_key'][:-1]

        return scan_info
        
    def _pull_rigid_scan(self, infile, current_line):
        """
        Pull the rigid scan information from the log file.

        Parameters
        ----------
        infile : :class:`iter`
            Lines of the log file.

        Returns
        -------
        :class:`list` of :class:`list of str` 
            List is a container for:
            initial_zmat : :class:`list of str`
                The initial Z matrix for the molecule.

            variables : :class:`list of str`
                Unprocessed lines of information for any set variables in
                the Z matrix.

        """
        # Initialise variables
        initial_zmat = []
        variables = []
        variables_flag = 'Variables'

        # File is ordered to give initial z matrix, a header 'Variables'
        # and then the variables with no blank lines in between.
        current_line = next(infile)

        # Pull all lines of the initial z matrix from the file
        while variables_flag not in current_line:
            initial_zmat.append(current_line.strip())
            current_line = next(infile)
        current_line = next(infile)

        # Pull all variables lines from file
        while current_line.strip() != '':
            variables.append(current_line.split())
            current_line = next(infile)

        return [initial_zmat, variables]

    def _process_rigid_scan(self, scan_input):
        """
        Process raw rigid scan input for each scan parameter.

        Parameters
        ----------
        scan_input : :class:`list` of :class:`list of str`
            Container of the initial Z matrix for the molecule
            and the unprocessed lines of information for any 
            set variables in the Z matrix.

        Returns
        -------
        :class:`dict` of :class: `dicts`
            An entry for each scan parameter that the PES is a 
            function of. Where Key is the 
            variable name in the Z matrix and Value is a 
            dictionary of scan information of form:
            {
                param_key : :class:`str`
                    '-' seperated atom id + index of all atoms in scan
                    parameter e.g. 'H1-O2'.
                atom_inds : :class:`list of int`
                    indexes of the atoms in scan parameter
                num_steps : :class:`int` - number of scan steps
                step_size : :class:`float` - size of the scan step
            }

        """
        # Unpack and intialise variables.
        initial_zmat, variables = scan_input
        total_scan_steps = 1
        scan_info = {}

        # Iterate over the variable entries to locate scan parameters.
        for var_line in variables:
            if any('Scan' in var for var in var_line):
                # Set the number of steps and step size
                # for each scanned parameter
                num_steps = int(var_line[2])
                total_scan_steps *= (num_steps+1)
                step_size = float(var_line[3])
                scan_info[var_line[0]] = {
                    'num_steps': num_steps,
                    'step_size': step_size
                    }

        # Find zmatrix line containing each scan variable.
        for atom, atom_zmat in enumerate(initial_zmat):
            for scan_var in list(scan_info.keys()):
                if scan_var in atom_zmat:

                    # Set scan parameter and atom inds (python index)
                    # from the initial z matrix.
                    atom_inds = [atom]
                    param_key = self.atom_ids[atom] + str(atom+1)
                    atom_entry = atom_zmat.split()
                    for i in range(1, atom_entry.index(scan_var), 2):
                        index = int(atom_entry[i]) - 1
                        atom_inds.append(index)
                        param_key += f'-{self.atom_ids[index]}{atom_entry[i]}'

                    # Set the scan parameter key and atom inds for each
                    # variable in the scan_info dict.
                    scan_info[scan_var]['atom_inds'] = atom_inds
                    scan_info[scan_var]['param_key'] = param_key

        return scan_info

    def pull_scan_input(self):
        """
        Get the relaxed/rigid scan information from the log file.

        Returns
        -------
        :class:`dict` of :class:`dict`
            An entry for each scan parameter that the PES is a 
            function of. Where Key is the scan variable name and
            Value is a dictionary of scan information of form:
            {
                param_key : :class:`str`
                    '-' seperated atom id + index of all atoms in scan
                    parameter e.g. 'H1-O2'.
                atom_inds : :class:`list` of :class:`int`
                    Indexes of the atoms in the scan parameter.
                num_steps : :class:`int`
                    The number of scan steps.
                step_size : :class:`float`
                    The size of the scan step.
            }

        """
        # Initialise functions and parse flag from scan type.
        if self.job_type == 'scan_rigid':
            parse_flag = 'Charge'
            pull_function = self._pull_rigid_scan
            process_function = self._process_rigid_scan
        else:
            parse_flag = 'ModRedundant input section'
            pull_function = self._pull_relaxed_scan
            process_function = self._process_relaxed_scan
        
        # Iterate through file and parse scan info section.
        with open(self.file_name, 'r') as infile:
            for line in infile:
                if parse_flag in line:
                    raw_scan_input = pull_function(line, infile)
                    break

        return process_function(raw_scan_input)

    def pull_atom_number(self):
        """
        Pull the number of atoms from start of log file.

        Alternative method if not normal termination.

        Returns
        -------
        :class:`int`
            The number of atoms in the molecule.

        """
        # Iterate through file and pull number of atoms.
        with open(self.file_name, 'r') as infile:
            for line in infile:
                if 'natoms' in line.lower():
                    return int(line.split()[1])

    def pull_multiplicity(self):
        """
        Pull the multiplicity from log file.
        
        Returns
        -------
        :class:`int` - multipliicity of molecule.
        
        """
        # Iterate over file and pull modredundant section.
        with open(self.file_name, 'r') as infile:
            for line in infile:
                if 'multiplicity' in line.lower():
                    return int(line.split()[-1])

    def pull_dipole_moment(self):
        """
        Pull x, y, z components and total of dipole moment from log file.

        Returns
        -------
        :class:`numpy.ndarray`
            A ``(1, 4)`` array of x, y, z components and 
            the total dipole moment of the molecule.
        
        """
        # Iterate over file and pull modredundant section.
        with open(self.file_name, 'r') as infile:
            for line in infile:
                if 'Dipole moment' in line:
                    input_line = infile.__next__().split()
                    dipole = np.asarray([float(input_line[i]) 
                                       for i in range(1,9,2)])
        return dipole

    def pull_trajectory(self, calculation_steps=None, opt=True):
        """
        Pull energy and geometry from intermediate calculation step(s).

        Can be used to pull intermediate or all steps from an
        optimisaton trajectory, rigid scan or relaxed scan.
        
        Parameters
        ----------
        calculation_steps : :class:`iterable` of :class:`ints`
            Target calculation steps. 
            [Default: None] If ``None`` then calculates all steps.
            Can be single `int` if single calculation step wanted.

        opt : :class:`bool`
            [Default: True]
            If ``True`` then searches for optimised geometries as
            calculation steps (e.g. relaxed scan).
            Else searches for SCF Done energies as calculation
            steps (e.g. opt trajectory, rigid scan).

        Returns
        -------
        step_results : :class:`dict` of :class:`dicts`           
            Where Value is dict of energies and geometries and
            Key is calculation step.

        """
        # Process target calculation step input.
        if calculation_steps is None:
            all_steps = True
            calculation_steps = [1]
        else:
            all_steps = False
        if isinstance(calculation_steps, int):
            calculation_steps = [calculation_steps]

        # Set starting count for calculation steps.
        calc_step_count = 0
        calc_step_ind = 0
        step_result = {}
        results = {}
        
        # Functions and flags for collecting energy and geometry.
        pull_functions = {
            'energy': self._pull_energy,
            'geom': self._pull_geometry,
            'opt': self._pull_optimised
        }
        prop_flags = {
            'energy': 'SCF Done',
            'geom': 'Standard orientation',
            'opt': 'Optimized Parameters'
        }
        
        # Iterate over log file and parse properties.
        with open(self.file_name, 'r') as infile:
            for line in infile:
                for prop, flag in prop_flags.items():
                    if flag in line:
                        step_result[prop] = pull_functions[prop](
                            infile, line)
                        
                        # Update step count depending on target.
                        if (all([prop == 'opt', opt]) or
                           all([prop == 'energy', not opt])):
                            calc_step_count += 1
                
                # Save target calculation step result.
                if (calc_step_count == calculation_steps[calc_step_ind]):
                    calc_step_ind += 1
                    results[calc_step_count] = step_result
                    step_result = {}

                    if all_steps:
                        calculation_steps.append(calc_step_count + 1)
                    elif calc_step_ind == len(calculation_steps):
                        return results

        return results
