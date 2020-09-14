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
