"""Module containing parsing functions for G16 log files."""

import numpy as np
from molLego.utilities.utils import readlines_reverse, parse_mol_formula

# Dict mapping job type to the properties contained in the log file
__job_to_property__ = {
    'opt': ['energy', 'geom', 'opt'],
    'freq': ['energy', 'geom', 'thermo', 'opt'],
    'fopt': ['energy', 'geom', 'thermo', 'opt'],
    'sp': ['energy', 'geom'],
    'scan_relaxed': ['energy', 'geom', 'opt'],
    'scan_rigid': ['energy', 'geom']
    }

    # Additional ones to add are trajectory; scan as property?

# Dict of search flags in log file for each property
__property_flags__ = {
    'energy': 'SCF Done',
    'geom': 'Standard orientation',
    'opt': 'Optimized Parameters',
    'thermo': 'Thermochemistry'
    }

class LogFileError(Exception):
    """Rasied when error in reading log file."""

class GaussianLog():
    """
    Represents a Gaussian log file.

    Attributes
    ----------
    atom_ids : :class:`list of str`
        The IDs of the atoms in the molecule.

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
    
    def __init__(self, log_file):
        """
        Initialise from Gaussian log file.

        Parameters
        ----------
        log_file : :class:`str`
            The name/path to the parent log file.

        """
        self.file_name = log_file
        self.normal_termination = self._check_normal_termination()

        # Use end of file log for normal termination:
        if self.normal_termination:
            output, job_input = self._pull_end_output()
            output = output.split('\\')

            # Set job details.
            # self.job_type = output[3]
            self.method, self.basis_set = output[4:6]

            # Set molecule details.
            atom_number, elements, charge = parse_mol_formula(output[6])
            self.atom_number = atom_number
            self.elements = elements
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
            
                # Set MP2 as method if present for correct energy parsing
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

    def _pull_flags(self):
        """
        Convert the job type to flags for seeking information in the log file.

        Returns
        -------
        job_property_flags : :class:`dict`
            A :class:`dict`, where the key is a property type and the value
            is the corresponding string flag for parsing the property from
            the Gaussian log file.

        """
        # Dict mapping job type to the properties contained in the log file
        job_to_property = {
            'opt': ['energy', 'geom', 'opt'],
            'freq': ['energy', 'geom', 'thermo', 'opt'],
            'fopt': ['energy', 'geom', 'thermo', 'opt'],
            'sp': ['energy', 'geom'],
            'scan_relaxed': ['energy', 'geom', 'opt'],
            'scan_rigid': ['energy', 'geom']
            }

        # Dict of search flags in log file for each property
        property_flags = {
            'energy': 'SCF Done',
            'geom': 'Standard orientation',
            'opt': 'Optimized Parameters',
            'thermo': 'Thermochemistry'
            }

        # Set parser search  the correct property flags
        try:
            flag_keys = job_to_property[self.job_type.lower()]
            job_property_flags = {
                prop: property_flags[prop] for prop in flag_keys
                }
        except KeyError:
            print('Job type is not recognised')

        # Switch energy flag if method is MP2.
        if (hasattr(self, 'method')):
            if ('mp2' in self.method.lower()):
                job_property_flags['energy'] = 'EUMP2'

        return job_property_flags

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
            The xyz coordinates for each atom.

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

    def _update_opt_count(self, property):
        """
        Update the count of the current geometry number in the log file.

        Updates once count for each optimised geometry if the calculation is an
        optimisation, or updates for each new geometry if the calculation is a
        rigid scan (single point).

        Parameters
        ----------
        property : :class:`str`
            The property being currently parsed from the log file

        Returns
        -------
        :class:`Bool`
            ``True`` if updating the opt count. ``False``, otherwise.

        """
        if property == 'energy':
            return self.spe
        elif property == 'opt':
            return (not self.spe)
        else:
            return False

    def get_properties(self, opt_steps=[1]):
        """
        Get properties from the log file.

        Parse properties such as the energy, thermodynamic data, geometry,
        and optimised information from specified optimsation step(s) in the log
        file for the molecule.

        Parameters
        ----------
        opt_steps : :class:`list of int`, optional
            Target optimisation/geometry step(s) wanted from the scan (rigid
            scan `spe=True``; relaxed scan `spe=False`) or optimisation
            trajectory (``spe=True``).

        Returns
        -------
        :class:`dict of dicts`
            A dictionary containing target properties for the molecule with
            the key as the optimisation/sp count [opt_steps].

        """
        # Initialise variables.
        step_result = {}
        opt_count = 0
        opt_step_ind = 0
        mol_results = {}

        # Mapping of functions to property.
        pull_functions = {
            'energy': self._pull_energy,
            'geom': self._pull_geometry,
            'thermo': self._pull_thermo,
            'opt': self._pull_optimised
            }

        # Set alternate flag if MP2 energy.
        if self.job_property_flags['energy'] == 'EUMP2':
            pull_functions['energy'] = self._pull_mp2_energy

        # Set opt count to 2 if fopt calculation
        # as thermo occurs after opt count met.
        if self.job_type == 'fopt':
            opt_steps = [2] if opt_steps == [1] else opt_steps

        # Open and iterate through log file.
        with open(self.file_name, 'r') as infile:
            for line in infile:
                # Check for property in the line and set property value.
                for prop, flag in self.job_property_flags.items():
                    if flag in line:
                        step_result[prop] = pull_functions[prop](infile, line)
                        opt_count += self._update_opt_count(prop)

                # If target optimisation step is met append results.
                if (opt_count == opt_steps[opt_step_ind]):
                    opt_step_ind += 1
                    mol_results[opt_count] = step_result
                    step_result = {}

                    # Return results if calculated for all optimisation steps.
                    if opt_step_ind == len(opt_steps):
                        return mol_results

    def _process_scan_info(self, scan_input):
        """
        Process scan information from modredundant input.

        Parameters
        ----------
        scan_input : `str`
            Unprocessed modredundant input line from log file.

        Returns
        -------
        :class:`dict`
            Details of the relaxed scan in a format:
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
        # Initialise variables.
        scan_info = {
            'param_key': '',
            'atom_inds': []
            }
        
        # Process scan information from modredundant input
        scan_input = scan_input.split()
        scan_info['num_steps'] = int(scan_input[-2])
        scan_info['step_size'] = float(scan_input[-1])

        # Set atom indexes (-1 for python index) and parameter key
        # ('-' seperated, atom id, atom index)
        s_index = scan_input.index('S')
        for i in scan_input[1:s_index]:
            scan_info['atom_inds'].append(int(i) - 1)
            scan_info['param_key'] += (f'{self.atom_ids[int(i)-1]}{i}-')
        scan_info['param_key'] = scan_info['param_key'][:-1]
        
        return scan_info

    def pull_scan_input(self):
        """
        Pull relaxed scan information from a log file.

        Returns
        -------
        :class:`dict`
            Scan information in the form:
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
        # Set modredundant input flag.
        scan_input = False
        modred_flag = 'The following ModRedundant input section has been read:'

        # Iterate over file and pull modredundant section.
        with open(self.file_name, 'r') as infile:
            for line in infile:
                if modred_flag in line:
                    modred_line = next(infile)
                    while modred_line.strip():
                        if 'S' in modred_line:
                            scan_input = self._process_scan_info(modred_line)
                        modred_line = next(infile)
                    
                    break
        return scan_input
        
    def _pull_rigid_scan(self, infile):
        """
        Pull the rigid scan information from the log file.

        Parameters
        ----------
        infile : :class:`iter`
            Lines of the log file.

        Returns
        -------
        :class:`list of str`
            The initial Z matrix for the molecule.

        :class:`list of str`
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

        return initial_zmat, variables

    def _process_rigid_scan(self, initial_zmat, variables):
        """
        Process raw rigid scan input for each scan parameter.

        Parameters
        ----------
        initial_zmat : :class:`list of str`
            The initial Z matrix for the molecule.

        variables : :class:`list of str`
            Unprocessed lines of information for any set variables in
            the Z matrix.

        Returns
        -------
        :class:`dict of dicts`
            An entry for each scan parameter where key is the variable name
            in the Z matrix and the value is a dictionary of form:
            {
                param_key : :class:`str`
                    '-' seperated atom id + index of all atoms in scan
                    parameter e.g. 'H1-O2'.
                atom_inds : :class:`list of int`
                    Indexes of the atoms in the scan parameter.
                num_steps : :class:`int`
                    The number of scan steps.
                step_size : :class:`float`
                    The size of the scan step.
            }

        """
        # Intialise variables
        total_scan_steps = 1
        scan_info = {}

        # Iterate over the variable entries to locate scan parameters
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

        # Find zmatrix line containing each scan variable
        for atom, atom_zmat in enumerate(initial_zmat):
            for scan_variable in list(scan_info.keys()):
                if scan_variable in atom_zmat:

                    # Set scan parameter and atom inds (python index)
                    # from the initial z matrix.
                    atom_inds = [atom]
                    param_key = self.atom_ids[atom] + str(atom+1)
                    atom_entry = atom_zmat.split()
                    for i in range(1, atom_entry.index(scan_variable), 2):
                        index = int(atom_entry[i]) - 1
                        atom_inds.append(index)
                        param_key += f'-{self.atom_ids[index]}{atom_entry[i]}'

                    # Set the scan parameter key and atom inds for each
                    # variable in the scan_info dict.
                    scan_info[scan_variable]['atom_inds'] = atom_inds
                    scan_info[scan_variable]['param_key'] = param_key

        return scan_info

    def get_rigid_scan_info(self):
        """
        Get the scan information for rigid scan from the log file.

        Returns
        -------
        :class:`dict of dicts`
            An entry for each scan parameter where key is the variable name
            in the Z matrix and the value is a dictionary of form:
            {
                param_key : :class:`str`
                    '-' seperated atom id + index of all atoms in scan
                    parameter e.g. 'H1-O2'.
                atom_inds : :class:`list of int`
                    Indexes of the atoms in the scan parameter.
                num_steps : :class:`int`
                    The number of scan steps.
                step_size : :class:`float`
                    The size of the scan step.
            }

        """
        zmat_start_flag = 'Charge'

        # Iterate through file and pull relevant input sections
        with open(self.file_name, 'r') as input:
            for line in input:
                if zmat_start_flag in line:
                    variables, initial_zmat = self._pull_rigid_scan(input)
                    break

        return self._process_rigid_scan(variables, initial_zmat)

    def pull_dipole_moment(self):
        """Sets x, y, z and total dipole moment."""
        with open(self.file_name, 'r') as input:
            for line in input:
                if 'Dipole moment' in line:
                    input_line = input.__next__().split()
                    dipole_components = [float(input_line[i]) for i in range(1,7,2)]
                    dipole_total = float(input_line[-1])

        return dipole_components, dipole_total

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

        
