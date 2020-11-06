"""Potentially useful functions."""


def readlines_reverse(filename):
    """Read lines in file reverse order."""
    with open(filename) as qfile:
        qfile.seek(0, 2)
        position = qfile.tell()
        line = ''
        while position >= 0:
            qfile.seek(position)
            next_char = qfile.read(1)
            if next_char == "\n":
                yield line[::-1]
                line = ''
            else:
                line += next_char
            position -= 1
        yield line[::-1]


def parse_mol_formula(mol_formula):
    """
    Extract elements, number of atoms and charge from molecular formula.

    Notes
    -----
    If charge is present then appended on the end in ()
    Example:
        C4H9O(-1)

    Parameters
    ----------
    mol_formula: `str`
        The molecular formula to be parsed.

    Returns
    -------
    atom_number: `int`
        The number of atoms in the molecule.
    elements: `list of str`
        The list of elements present in the molecule.
    charge: `int`
        The charge of the molecule.

    """
    atom_number = 0
    elements = []

    # Identify charge and remove it from the molecular formula.
    if mol_formula.isalnum():
        charge = 0
    else:
        charge = int(mol_formula.split('(')[1][:-2])
        if mol_formula[-2] == '-':
            charge *= -1
    mol_formula = mol_formula.split('(')[0]

    # Find individual elements in the molecular formula.
    i = 0

    while i < len(mol_formula)-1:
        char = mol_formula[i]
        while char.isdigit() is mol_formula[i+1].isdigit():
            char += mol_formula[i+1]
            i += 1
        # Identify if an element or a count.
        if char.isdigit():
            atom_number += int(char)
        else:
            elements.append(char)
        i += 1
        # Add last entry if not already included
        if i != len(mol_formula):
            if mol_formula[-1].isdigit():
                atom_number += int(mol_formula[-1])
            else:
                elements.append(mol_formula[-1])

    return atom_number, elements, charge
