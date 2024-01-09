import itertools


def get_incar_magmom(structure):
    atoms = structure.get_ase()
    magmom = {
        "Ce": 5,
        "Co": 1,
        "Cr": 5,
        "Dy": 5,
        "Er": 3,
        "Eu": 10,
        "Fe": 5,
        "Gd": 7,
        "Ho": 4,
        "La": 0.6,
        "Lu": 0.6,
        "Mn": 5,
        "Mo": 5,
        "Nd": 3,
        "Ni": 5,
        "Pm": 4,
        "Pr": 2,
        "Sm": 5,
        "Tb": 6,
        "Tm": 2,
        "V": 5,
        "W": 5,
        "Yb": 1,
    }

    text = []
    for k, g in itertools.groupby(atoms.get_chemical_symbols()):
        text.append("{}*{}".format(len(list(g)), magmom.get(k, 1)))

    return {
        "MAGMOM": text,
    }


def get_incar_ldau(structure):
    atoms = structure.get_ase()
    ldau = {
        "Mn": 4,
        "Fe": 4.6,
        "Cu": 4,
        "Cd": 2.1,
        "Pd": 3.6,
    }
    elements = set(atoms.get_chemical_symbols())

    if elements & set(ldau):
        ldaul = []
        ldauu = []
        for k, g in itertools.groupby(atoms.get_chemical_symbols()):
            ldaul.append(2 if k in ldau else -1)
            ldauu.append(ldau[k] if k in ldau else 0)
        return {
            "LDAU": True,
            "LDAUTYPE": 2,
            "LDAUL": ldaul,
            "LDAUU": ldauu,
            "LDAUJ": [0] * len(elements),
        }

    else:
        return {}


def get_incar_spin(structure, magnetic_moment):
    if magnetic_moment < 0.5:
        return {}

    else:
        tags = {"ISPIN": 2}
        tags.update(get_incar_magmom(structure))
        return tags


def get_incar_dipole(structure, dipole_moment):
    atoms = structure.get_ase()

    if dipole_moment < 0.05:
        return {}

    else:
        return {
            "LDIPOL": True,
            "IDIPOL": 3,
            "DIPOL": atoms.get_center_of_mass(scaled=True),
        }
