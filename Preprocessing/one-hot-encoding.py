import pandas as pd

# Read the input file
df = pd.read_csv("Dihedrals_test_10.csv", header=0)

# Open the output file
with open("OHE_test_no-sec_10.csv", "a") as outfile:
    # All possible categories
    values = ["ALA", "ARG", "ASN", "ASP", "CYS", "E", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

    # Categorical columns
    categorical_cols = ['res', '-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    # Convert categorical columns and specify the possible categories
    for item in categorical_cols:
        df[item] = pd.Categorical(df[item], categories=values)

    # One-hot encode the categorical columns
    dummies = pd.get_dummies(df, columns=categorical_cols)

    # Write the one-hot encoded data to the output file
    dummies.to_csv(outfile, index=False)
