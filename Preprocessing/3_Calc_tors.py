import pandas as pd
import Bio.PDB

def calculate_dihedrals(data):
    output = []
    for i in range(1, data.shape[0]):
        if int(data.nr[i - 1]) + 1 != data.nr[i]:
            continue

        N = Bio.PDB.vectors.Vector(*data.loc[i, ['Nx', 'Ny', 'Nz']].values)
        N_1 = Bio.PDB.vectors.Vector(*data.loc[i + 1, ['Nx', 'Ny', 'Nz']].values)
        Ca_0 = Bio.PDB.vectors.Vector(*data.loc[i - 1, ['Cax', 'Cay', 'Caz']].values)
        Ca = Bio.PDB.vectors.Vector(*data.loc[i, ['Cax', 'Cay', 'Caz']].values)
        C_0 = Bio.PDB.vectors.Vector(*data.loc[i - 1, ['Cx', 'Cy', 'Cz']].values)
        C = Bio.PDB.vectors.Vector(*data.loc[i, ['Cx', 'Cy', 'Cz']].values)

        phi_rad = Bio.PDB.vectors.calc_dihedral(C_0, N, Ca, C)
        psi_rad = Bio.PDB.vectors.calc_dihedral(N, Ca, C, N_1)

        res = []
        for x in range(-10, 11):
            if i + x < 0 or i + x >= data.shape[0]:
                res.append('E')
            elif data.nr[i + x] - x == data.nr[i]:
                res.append(data.Res[i + x])
            else:
                res.append('E')

        output.append([data.nr[i], data.Res[i], *res, phi_rad, psi_rad])

    return pd.DataFrame(output, columns=['nr', 'res', *[f'{i}' for i in range(-10, 11)], 'phi', 'psi'])

def save_dihedrals(df, filename):
    df.to_csv(filename, index=False)

data = pd.read_csv('Coordinates.csv', header=0, low_memory=False)
dihedrals = calculate_dihedrals(data)
save_dihedrals(dihedrals, 'Dihedrals_10.csv')
