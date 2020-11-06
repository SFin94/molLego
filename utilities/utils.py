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

    If charge is present then appended on the end in () with sign last.
    Example format:
        C4H9O1(1-)

    Parameters
    ----------
    mol_formula : `str`
        The molecular formula to be parsed.

    Returns
    -------
    atom_number : `int`
        The number of atoms in the molecule.
    elements : `list of str`
        The list of elements present in the molecule.
    charge : `int`
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
    char = mol_formula[0]
    for i in mol_formula[1:]:
        # Check for switch in value type.
        if i.isdigit() != char.isdigit():
            # Identify if an element or a count.
            if char.isdigit():
                atom_number += int(char)
            else:
                elements.append(char)
            char = i
        else:
            char += i
    # Add last characters.
    if char.isdigit():
        atom_number += int(char)
    else:
        elements.append(char)

    return atom_number, elements, charge


