
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
        A ``(N, 3)`` array of x, y, z coorindates for each atom.
        Where N is the number of atoms in the molecule.
    
    e: :class:`float`
        The thermally corrected energy of the molecule.
    
    escf: :class:`float`
        The SCF Done energy of the molecule.
    
    h: :class:`float`
        The thermally corrected enthalpy of the molecule.
    
    g: :class:`float`
        The thermally corrected Gibbs free energy of the molecule.
    
    s: :class:`float`
        The entropy of the molecule.
    
    zpe: :class:`float`
        The Zero Point Energy of the molecule.
   
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

        # Set additional thermodynamic attributes.
        self.zpe = properties['thermo']['ZPE']
        self.e = properties['thermo']['E']
        self.h = properties['thermo']['H']
        self.g = properties['thermo']['G']
        self.s = properties['thermo']['S']
