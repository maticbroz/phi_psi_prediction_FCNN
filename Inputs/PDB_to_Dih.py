from Bio.PDB import *
import math
import pandas as pd
import numpy as np
import warnings

# Constants
AMINO_ACIDS = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
PDBS_LIST_PATH = "PDBs_list.csv"
NON_OHE_DATASET_PATH = "Non_OHE_dataset.csv"
PDBS_DIR = "All_pdbs/"

warnings.simplefilter('ignore', BiopythonWarning)

def pdb_to_dih(pdb, chain):
    """Calculates dihedral angles and creates a sliding window from a PDB that does not contain errors."""
    # Load PDB file
    parser = PDBParser()
    pdb_file = f"{PDBS_DIR}{pdb}.pdb"
    my_protein = parser.get_structure(pdb, pdb_file)
    my_chain = my_protein[0][chain]
    residues = list(my_chain.get_residues())
    residues = [res for res in my_chain.get_residues() if res.get_resname() in AMINO_ACIDS]

    for i in range(len(residues) - 1):
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
        phi = round(math.degrees(calc_dihedral(atom1, atom2, atom3, atom4)), 3)
        psi = round(math.degrees(calc_dihedral(atom2, atom3, atom4, atom5)), 3)   

        angle_data.append([pdb, residue.get_resname(), residue.get_id()[1], phi, psi, *window ])

    angles = pd.DataFrame(angle_data)
    return angles

# Read PDBs list from CSV
df_list = pd.read_csv(PDBS_LIST_PATH, header=0)

def loop(pdb, chain):
    try:
        # Append the angle data to the non OHE dataset CSV
        pdb_to_dih(pdb, chain).to_csv(NON_OHE_DATASET_PATH, index=False, mode="a", header=0)
    except:
        pass

# Loop over the PDBs list and apply the loop function to each row
df_list.apply(lambda x: loop(x['PDB'],
