
"""Module containing thermodynamic specific Molecule child classes."""

from molLego.parsers.parse_gaussian import GaussianLog
from molLego.molecules.molecule import Molecule

class GaussianThermoMolecule(Molecule):
    """
    Represents a Molecule from a Gaussian Frequency calculation.

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
        The SCF Done energy of the molecule (a.u.).

    e_therm : :class:`float`
        The thermally corrected energy of the molecule (kJ/mol).
    
    h : :class:`float`
        The thermally corrected enthalpy of the molecule (kJ/mol).
    
    g : :class:`float`
        The thermally corrected Gibbs free energy of the molecule
        (kJ/mol).
    
    s : :class:`float`
        The entropy of the molecule (kJ/mol).

    t : :class:`float`
        The temperature the properties are calculated at.
    
    zpe: :class:`float`
        The Zero Point Energy of the molecule (kJ/mol).
   
    """

    def __init__(self, output_file, parser=GaussianLog):
        """
        Initialise a Molecule from Gaussian Frequency calculation.

        Parameters
        ----------
        parser: `OutputParser`
            Parser to use for calculation output.
            [default: GaussianLog]
        
        output_file: `str`
            The path to the calculation output file.

        """
        # Set parser object.
        self.parser = parser(output_file)

        # Get base properties from calculation parser.
        properties = self.parser.get_properties()

        # Set attributes.
        self.atom_ids = self.parser.atom_ids
        self.atom_number = len(self.atom_ids)
        self.charge = properties['charge']
        self.geometry = properties['geom']
        self.e = properties['energy']
        self.parameters = {}

        # Set additional thermodynamic attributes.
        self.zpe = properties['thermo']['zpe']
        self.e_therm = properties['thermo']['e_therm']
        self.h = properties['thermo']['h']
        self.g = properties['thermo']['g']
        self.s = properties['thermo']['s']
        self.t = properties['thermo']['t']

    def get_df_repr(self):
        """
        Create dict representation of Molecule for a DataFrame.

        Returns
        -------
        df_rep : `dict`
            Molecule properties in the format:
            {
                file_name   : path to parent output file,
                e : SCF electronic energy (kJ/mol),
                e_therm : thermally corrected energy (kJ/mol),
                zpe : zero point energy (kJ/mol)
                h : thermally corrected enthalpy (kJ/mol)
                s : entropy (kJ/mol)
                g : thermally corrected Gibbs free energy (kJ/mol)
                (optional)
                parameter key : parameter value
                [for all parameters in self.parameters]
            }

        """
        df_rep = {'file_name': self.parser.file_name,
                  'e': self.e,
                  'e_therm': self.e_therm,
                  'zpe': self.zpe,
                  'h': self.h,
                  's': self.s,
                  'g': self.g}
        df_rep.update(self.parameters)

        return df_rep
