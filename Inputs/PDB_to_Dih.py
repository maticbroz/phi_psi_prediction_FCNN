from Bio.PDB import *
import math
import pandas as pd
import numpy as np
from sys import exit
from Bio import BiopythonWarning
import warnings

warnings.simplefilter('ignore', BiopythonWarning)

amino_acids = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]

""" Calculates dihedral angles and creates a sliding window from a PDB that does not contain errors."""

def pdbToDih(pdb, chain):

    # Load PDB file
    parser = PDBParser()
    pdb_file = f'All_pdbs/{pdb}.pdb'
    myProtein = parser.get_structure(pdb, pdb_file)
    myChain = myProtein[0][chain]
    residues = list(myChain.get_residues())
    residues = [res for res in myChain.get_residues() if res.get_resname() in amino_acids]

    for i in range(len(residues)-1):
        if residues[i].get_id()[1] + 1 != residues[i+1].get_id()[1]:
            return

    angle_data = []
    # Loop through the residues in the chain
    num_res = len(residues)
    for i, residue in enumerate(residues):
        window = np.full((21,), "EEE", dtype=np.dtype((str, 3)))
        # Skip the first residue (N-terminus)
        if i == 0 or i == num_res - 1:
            continue

        for j in range(-10, 11):
            if (i + j >= 0) & (i + j + 1 < num_res):
                window[j+10] = residues[i+j].get_resname()        

        # Get the atoms that form the dihedral angle
        atom1 = Vector(residues[i-1]['C'].coord)
        atom2 = Vector(residue['N'].coord)
        atom3 = Vector(residue['CA'].coord)
        atom4 = Vector(residue['C'].coord)
        atom5 = Vector(residues[i+1]['N'].coord)   

        # Calculate the dihedral angle using the calc_dihedral function
        phi = round(math.degrees(calc_dihedral(atom1, atom2, atom3, atom4)),3)
        psi = round(math.degrees(calc_dihedral(atom2, atom3, atom4, atom5)),3)   

        angle_data.append([pdb, residue.get_resname(), residue.get_id()[1], phi, psi, *window ])

    angles = pd.DataFrame(angle_data)
    return angles

df_list = pd.read_csv("PDBs_list.csv", header=0)

def loop(pdb, chain):
    try:
        pdbToDih(pdb, chain).to_csv('Non_OHE_dataset.csv', index=False, mode="a", header=0)
    except: pass

df_list.apply(lambda x: loop(x['PDB'], x['Chain']), axis=1)
