
"""Module containing trajectory/PES specific Molecule child classes."""

import molLego.utilities.geom as geom
from molLego.molecules.molecule import Molecule

class PESStep():
    """
    Represents a single point on a molecules PES.

    Attributes
    ----------
    e : :class:`float`
        The energy value of the geometry arrangement.

    geometry : :class:`numpy ndarray`
        A ``(N, 3)`` array of x, y, z coordinates for each atom.
        Where N is the number of atoms in the molecule.

    opt : :class:`bool`
        ``True`` if the step is optimised,
        ``False`` otherwise.
        [Default: False]

    parameters : :class:`dict`
        Where Key is the parameter identifier,
        and Value is the geometric parameter.
        E.g. O1-C2-O3: 180.00

    step_id : :class:`int`
        The step number on the PES.    
        
    """
    
    def __init__(self, step_id, geometry, e, opt=False):
        """
        Initialise step in PES.

        Parameters
        ----------
        step_id : :class:`int`
            The step number on the PES. 

        geometry : :class:`numpy ndarray`
            A ``(N, 3)`` array of x, y, z coordinates for each atom.
            Where N is the number of atoms in the molecule.
        
        e : :class:`float`
            The energy value of the geometry arrangement.
        
        opt : :class:`bool`
            ``True`` if the step is optimised,
            ``False`` otherwise.
            [Default: False]            

        """
        self.step_id = step_id
        self.geometry = geometry
        self.e = e
        self.opt = opt
        self.parameters = {}

    def set_parameters(self, params, param_keys):
        """
        Calculate geometric parameters (bonds, angles, dihedrals).

        Parameters saved in `dict`.
        Where Key is `str` of each atom_id and atom_index.
        E.g. C1-H2
        Value is the value of the geometric parameter.

        Parameters
        ----------
        params : :class:`iterable` of :class:`int`
            The atom indexes defining the geometric parameter.
        
        param_keys : :class:`list` of `str`
            The parameter identifier in the form of the atom
            id and index for each atom in the parameter.

        """
        # Calculate parameter values.
        for par in params:
            param_vals = geom.calc_param(par, self.geometry)
        
        # Update parameter dict.
        self.parameters.update(dict(zip(param_keys, param_vals)))

    def get_df_repr(self):
        """
        Create dict representation of Molecule for a DataFrame.

        Returns
        -------
        df_rep : `dict`
            PES step properties in the format:
            {
                step_id : step number on PES
                e : energy (kJ/mol),
                opt : `bool`, whether the point is optimised
                parameter key : parameter value
                [for all parameters in self.parameters]
            }

        """
        df_rep = {'step_id': self.step_id,
                  'e': self.e,
                  'opt': self.opt}
        df_rep.update(self.parameters)

        return df_rep


class PESMolecule(Molecule):
    """
    Represents a Molecule with a PES/trajectory.

    Child class of Molecule.

    Attributes
    ----------
    atom_ids : :class:`list of str`
        The atomic symbols of the atoms in the molecule.

    atom_number : :class:`int`
        The number of atoms in the molecule.
    
    charge : :class:`int`
        The formal charge of the molecule.
    
    geometry : :class:`numpy ndarray`
        A ``(N, 3)`` array of x, y, z coordinates for each atom.
        Where N is the number of atoms in the molecule.
    
    e : :class:`float`
        The energy of the molecule.

    parser : `OutputParser`
        Parser to use for calculation output.

    pes : :class:`list` of :PESStep:
        A PESStep representing each each point on the PES.

    scan_info : :class:`dict` of :class:`dict`
            An entry for each scan parameter that the PES is a 
            function of. Where Key is the scan variable name and
            Value is a dictionary of scan information of form:
        {
            param_key : :class:`str`
                '-' seperated atom id + index of all atoms in scan
                parameter e.g. 'H1-O2'.
            atom_inds : :class:`list` of :class:`int`
                indexes of the atoms in scan parameter
            num_steps : :class:`int` - number of scan steps
            step_size : :class:`float` - size of the scan step
        }

    """

    def __init__(self, output_file, 
                 parser, 
                 calculation_steps=None):
        """
        Initialise a Molecule with a PES.

        Parameters
        ----------        
        output_file : `str`
            The path to the calculation output file.

        parser : `OutputParser`
            Parser to use for calculation output.
            [Default: GaussianLog]

        calculation_steps : :class:`iterable` of :class:`ints`
            Target calculation steps. 
            [Default: None] If ``None`` then calculates all steps.
            Can be single `int` if single calculation step wanted.

        """
        # Set inherited properties.
        super().__init__(output_file, parser)
        
        # Set scan information.
        scan_steps = 1
        self.scan_info = self.parser.pull_scan_input()
        for scan in self.scan_info.values():
            scan_steps *= scan['num_steps'] + 1
        
        # Set calculation steps to match scan if not specified.
        if calculation_steps is None:
            calculation_steps = list(range(1, scan_steps+1))

        # Parse PES and set PES steps.
        opt = (self.parser.job_type != 'scan_rigid')
        pes_full = self.parser.pull_trajectory(calculation_steps, opt)
        self.pes = []
        for step, results in pes_full.items():
            self.pes.append(PESStep(step, results['geom'],
                            results['energy'], results['opt']))

        # Calculate parameters for scan parameters.
        scan_params = [x['atom_inds'] for x in self.scan_info.values()]
        self.set_parameters(scan_params)

    def set_parameters(self, params):
        """
        Calculate geometric parameters for each PES Step.

        The parameters are set as attributes for each of the PESSteps
        in the PES of the molecule.

        Parameters
        ----------
        params : :class:`iterable` of :class:`int`
            The atom indexes defining the geometric parameter.
            Can be either a nested iterable is multiple parameters,
            or a single iterable if only one parameter is required.

        """
        # Handle if single parameter.
        if not any(isinstance(x, (list, tuple)) for x in params):
            params = [params]

        # Set parameter keys.
        param_keys = []
        for par in params:
            # Set parameter key as atom_ID+atom_index'-'atom_ID+atom_index [-...-...]
            param_keys.append('-'.join([self.atom_ids[i] + str(i) for i in par]))
        
        # Calculate parameter value for each step on PES.
        for pes_step in self.pes:
           pes_step.set_parameters(params, param_keys)  

    def get_pes_step(self, step_index=None):
        """
        Yield the potential energy steps of the molecule.

        Parameters
        ----------
        step_index : :class: `iterable` of :class:`int`
            The index(es) of the PES steps wanted.
            [default: ``None``] If ``None`` then returns all PES steps.
            Can be single `int` to call single PES step.

        Yields
        -------
        :class: :PESStep:
            PES steps of molecule.

        """
        # Set to all atoms is atom_index is None.
        if step_index is None:
            step_index = range(len(self.pes))
        elif isinstance(step_index, int):
            step_index = (step_index, )

        for step in step_index:
            yield self.pes[step]

    def get_df_repr(self):
        """
        Yield dict representation of each PES step for a DataFrame.

        Yields
        -------
        df_rep : `dict`
            Molecule properties for each PES step format:
            {
                file_name : path to parent output file,
                step_id : step number on PES
                e : energy (kJ/mol),
                opt : `bool`, whether the point is optimised
                parameter key : parameter value
                [for all parameters in self.parameters]
            }

        """
        for step in self.get_pes_step():
            df_rep = {'file_name': self.parser.file_name}
            df_rep.update(step.get_df_repr())
            yield df_rep
