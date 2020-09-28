"""Module containing parsing functions for G16 log files."""

import numpy as np
from .utils import readlines_reverse, parse_mol_formula


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

    method : :class:`str`
        The method of the calculation (functional, etc.).

    job_property_flags : :class:`dict`
        A :class:`dict`, where the key is a property type and the value
        is the corresponding string flag for parsing the property from
        the Gaussian log file.

    job_type : :class:`str`
        The type of the calculation (Opt, Fopt, Freq, SP, Scan).

    normal_termination : :class:`Bool`
        ``True`` if normal termination. Otherwise, ``False``.

    spe : :class:`bool`
        If ``True``, the calculation is a single point energy calculation.
        Otherwise, the calculation is a result of an optimisation.

    """

    def __init__(self, logfile):
        """
        Initialise from Gaussian log file.

        Parameters
        ----------
        logfile : :class:`str`
            The path to the parent log file.

        """
        self.file_name = logfile
        self.normal_termination = self._check_normal_termination()
        extra_output = self._pull_extra_output()

        # Use end of file log for normal termination:
        if self.normal_termination:
            output = self._pull_end_output()

            self.job_type = output[3].lower()
            self.method, self.basis_set = output[4:6]
            atom_number, elements, charge = parse_mol_formula(output[6])
            self.atom_number = atom_number
            self.elements = elements
            self.charge = charge
            self.atom_ids = self._pull_atom_ids()

            # Set scan type.
            if self.job_type.lower() == 'scan':
                if extra_output['modredundant']:
                    self.job_type = 'scan_relaxed'
                else:
                    self.job_type = 'scan_rigid'

            print(
                'Normal termination output found, '
                'all attributes set.'
                )

        # Try to use the beginning of the file job input data:
        else:
            try:
                output = self._pull_start_output()

                # Use flags to try to deduce the job type.
                calculation_flags = {
                    'opt': False,
                    'freq': False,
                    'scan': False,
                    'mp2': False
                    }
                job_calculation_types = {
                    'opt': [True, False, False],
                    'fopt': [True, True, False],
                    'freq': [False, True, False],
                    'scan_rigid': [False, False, True]
                    }

                # Set calculation flag to True if present in job input.
                for flag in calculation_flags.keys():
                    if flag in output.lower():
                        calculation_flags[flag] = True

                # Set job type based on calculation bool results.
                for job, calc_type in job_calculation_types.items():
                    if calc_type == list(calculation_flags.values())[:-1]:
                        self.job_type = job
                        break

                # If both scan and modredundant flags are true
                # then job is a relaxed scan.
                if all(extra_output.values()):
                    self.job_type = 'scan_relaxed'

                # Set MP2 as method if present for correct energy parsing
                if calculation_flags['mp2']:
                    self.method = 'mp2'

                # Set number of atoms and atom ids
                with open(self.file_name, 'r') as infile:
                    for line in infile:
                        if 'natoms' in line.lower():
                            self.atom_number = int(line.split()[1])
                            break
                self.atom_ids = self._pull_atom_ids()

                print(
                    'Normal termination output not present, '
                    'fewer attributes set using input information.'
                    )

            except:
                print('Cannot parse the job information from the log file.')

        # Set spe flag if single point and no optimisation.
        self.spe = any([
            (self.job_type.lower() == 'sp'),
            (self.job_type.lower() == 'scan_rigid')
            ])

        self.job_property_flags = self._pull_flags()

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

    def _pull_extra_output(self):
        extra_info_flags = {
            'modredundant': False,
            'scan ': False
            }

        with open(self.file_name, 'r') as infile:
            for line in infile:
                for flag in extra_info_flags.keys():
                    if flag in line.lower():
                        extra_info_flags[flag] = True

        return extra_info_flags

    def _pull_end_output(self):
        output = ''
        with open(self.file_name, 'r') as infile:
            line = next(infile)
            # Skip to the end log:
            while '1\\1\\' not in line:
                line = next(infile)
            # Pull the end log ouput:
            while '\\\\' not in line:
                output += line.strip()
                line = next(infile)
        return output.split('\\')

    def _pull_start_output(self):
        output = ''
        with open(self.file_name, 'r') as infile:
            line = next(infile)
            # Skip to the start log:
            while '%mem=' not in line:
                line = next(infile)
            while '---' not in line:
                line = next(infile)
            line = next(infile)
            # Pull the start log :
            while '---' not in line:
                output += line.strip()
                line = next(infile)
        return output

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

        # Try to pull the correct property flags
        try:
            flag_keys = job_to_property[self.job_type.lower()]
            job_property_flags = {
                prop: property_flags[prop] for prop in flag_keys
                }
        except KeyError:
            print('Job type is not recognised')

        # Switch energy flag if method is MP2.
        # Extra handle in case method was not parsed
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
        # Initialise variables
        atom_ids = []
        atom_id_flag = 'Charge = '
        jump_line_flags = [
            'No Z-Matrix',
            'Redundant internal coordinates'
            ]

        # Iterate through file and pull final section
        with open(self.file_name, 'r') as infile:
            for line in infile:
                if atom_id_flag in line:
                    line = next(infile)
                    if any(flag in line for flag in jump_line_flags):
                        line = next(infile)
                    for _ in range(self.atom_number):
                        atom_ids.append(line.split()[0][0])
                        line = next(infile)

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
        numpy array
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
                float(line.split()[jind+3])
                for jind in range(3)
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

        # Optional section if thermal corrections to E, H, G wanted
        # thermal_corrections = [float(next(input)[50:58]) for i in range(3)]
        # return thermal_corrections

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
            scan 1`spe=True``; relaxed scan `spe=False`) or optimisation
            trajectory (``spe=True``).

        Returns
        -------
        :class:`dict of dicts`
            A dictionary containing target properties for the molecule with
            the key as the optimisation/sp count [opt_steps]

        """
        # Initialise variables
        step_result = {}
        opt_count = 0
        opt_step_ind = 0
        mol_results = {}

        # Mapping of functions to property
        pull_functions = {
            'energy': self._pull_energy,
            'geom': self._pull_geometry,
            'thermo': self._pull_thermo,
            'opt': self._pull_optimised
            }

        if self.job_property_flags['energy'] == 'EUMP2':
            pull_functions['energy'] = self._pull_mp2_energy

        # Set opt count to 2 if fopt calculation
        # as thermo occurs after opt count met
        if any([self.job_type == 'fopt', self.job_type == 'freq']):
            opt_steps = [2] if opt_steps == [1] else opt_steps

        # Open and iterate through log file
        with open(self.file_name, 'r') as infile:
            for line in infile:
                # Check if any property is in the line and set property value
                for prop, flag in self.job_property_flags.items():
                    if flag in line:
                        step_result[prop] = pull_functions[prop](infile, line)
                        opt_count += self._update_opt_count(prop)

                # If target optimisation step is met append results
                if (opt_count == opt_steps[opt_step_ind]):
                    opt_step_ind += 1
                    mol_results[opt_count] = step_result
                    step_result = {}

                    # Return if results calculated for all optimisation steps
                    if opt_step_ind == len(opt_steps):
                        return mol_results

    def _pull_modredundant(self, infile):
        """
        Pull the scan information from the log file.

        Parameters
        ----------
        infile : :class:`iter`
            Lines of the log file.

        Returns:
        :class:`list of str`
            List of all moredundant input lines from log file.

        """
        # Goes to start of modredundant section,
        # then extracts all modredundant lines.
        modred_input = []
        current_line = next(infile)
        while current_line.strip() != '':
            modred_input.append(current_line.strip().split())
            current_line = next(infile)
        return modred_input

    def _process_modredundant(self, modred_input):
        """
        Process the scan information from the modredundant input.

        Parameters
        ----------
        modred_input : :class:`list of list of str`
            Unprocessed modredundant input lines from log file, as returned by
            the `_pull_modredundant()` method.

        Returns
        -------
        :class:`dict`
            Details of the scan in a format:
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
        # Initialise variables
        scan_info = {
            'param_key': '',
            'atom_inds': []
            }

        # Locates scan input line from modredundant input
        # by finding 'S' character
        for modred_line in modred_input:
            if 'S' in modred_line:
                scan_input = modred_line
                break

        # Process scan information from modredundant input
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

    def get_scan_info(self):
        """
        Get the scan information for relaxed scan from a log file.

        Returns:
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
        # Set modredundant information flag
        modred_flag = 'The following ModRedundant input section has been read:'

        # Iterate over file and pull modredundant section
        with open(self.file_name, 'r') as infile:
            for line in infile:
                if modred_flag in line:
                    modred_input = self._pull_modredundant(infile)
                    break

        # Process scan information from logfile modredundant input
        return self._process_modredundant(modred_input)

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
