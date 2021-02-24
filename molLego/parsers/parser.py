"""Module containing parent Parser class for output files."""
class OutputParser():
    """
    Parser base class for calculation output file.

    Has to set atom_ids, charge, 
    geometry and energy for molecule.

    """
    
    def __init__(self, output_file):
        raise NotImplementedError

    def _check_normal_termination(self):
        """Check if calculation terminated correctly."""
        raise NotImplementedError

    def _pull_atom_ids(self):
        """Pull the atom IDs from calculation output file."""
        raise NotImplementedError

    def _pull_charge(self):
        """Pull the energy from calculation output file."""
        raise NotImplementedError

    def _pull_energy(self):
        """Pull the energy from calculation output file."""
        raise NotImplementedError

    def _pull_geometry(self):
        """Pull the geometry from calculation output file."""
        raise NotImplementedError

    def get_properties(self):
        """Get all properties from calculation output file."""
        raise NotImplementedError
