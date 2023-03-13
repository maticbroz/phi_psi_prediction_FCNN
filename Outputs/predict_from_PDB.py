import warnings
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import BiopythonWarning
from Bio.PDB import *
from keras import backend as K
import tensorflow as tf

warnings.simplefilter('ignore', BiopythonWarning)

def PDBPredict(pdb):

    model_phi = tf.keras.models.load_model("save_PHI_latest", compile=False)
    model_psi = tf.keras.models.load_model("save_PSI_latest", compile=False) 

    amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

    parser = PDBParser()
    pdb_file = f'{pdb}.pdb'
    myProtein = parser.get_structure(pdb, pdb_file)
    myChain = myProtein[0]
    residues = [res for res in myChain.get_residues() if res.get_resname() in amino_acids]

    angle_data = []
    num_res = len(residues)
    for i, residue in enumerate(residues):
        window = np.full((21,), "EEE", dtype=np.dtype((str, 3)))
        if i == 0 or i == num_res - 1:
            continue

        for j in range(-10, 11):
            if (i + j >= 0) & (i + j + 1 < num_res):
                window[j+10] = residues[i+j].get_resname()

        atom1 = Vector(residues[i-1]['C'].coord)
        atom2 = Vector(residue['N'].coord)
        atom3 = Vector(residue['CA'].coord)
        atom4 = Vector(residue['C'].coord)
        atom5 = Vector(residues[i+1]['N'].coord)   

        phi = round(math.degrees(calc_dihedral(atom1, atom2, atom3, atom4)), 3)
        psi = round(math.degrees(calc_dihedral(atom2, atom3, atom4, atom5)), 3)   

        angle_data.append([residue.get_resname(), residue.get_id()[1], phi, psi, *window])

    angles = pd.DataFrame(angle_data, columns=["Res", "Num", "Phi", "Psi", *range(-10, 11)])
    elements = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "EEE", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    categorical_cols = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for item in categorical_cols:
        angles[item] = pd.Categorical(angles[item], categories=elements)

    angles = pd.get_dummies(angles, columns=categorical_cols)
    data = angles.drop(columns=angles.filter(regex='EEE').columns)

    def custom_loss(y_true, y_pred):
        D  = K.abs(y_true - y_pred)
        AE = K.minimum(D, K.abs(360 - D))
        return AE
    
    def adjust(angle):
        angle = (angle + 180) % 360 - 180
        return round(angle, 3)

    def diff(y_real, y_pred):
        D = abs(y_real - y_pred)
        AE = min(D, abs(360 - D))
        return round(AE, 3) 

    model_phi.compile(loss=custom_loss)
    model_psi.compile(loss=custom_loss)

    input = data.iloc[:, 4:]

    data['Phi_pred'] = model_phi.predict(input)
    data['Psi_pred'] = model_psi.predict(input)

    data['Phi_pred'] = data['Phi_pred'].apply(adjust)
    data['Phi_diff'] = data.apply(lambda x: diff(x['Phi'], x['Phi_pred']), axis=1)

    data['Psi_pred'] = data['Psi_pred'].apply(adjust)
    data['Psi_diff'] = data.apply(lambda x: diff(x['Psi'], x['Psi_pred']), axis=1)

    print(f"PDB {pdb}\nPhi diff: {data['Phi_diff'].mean().__round__(2)}\nPsi diff: {data['Psi_diff'].mean().__round__(2)}")