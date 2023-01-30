import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../Outputs/Output.csv", header=0)
number_of_rows = len(df.index)
cm = 1/2.54

fig, axes = plt.subplots(ncols=1, nrows=20, figsize=(32*cm,480*cm))

for i, ax in enumerate(axes.flatten()):
    neighbour_res = i - 10 if i > 9 else i - 9
    neighbour_res = str(neighbour_res)

    grouping = df.groupby(['res', neighbour_res])['psi_diff_adj'].mean().reset_index()
    grouped = grouping.pivot(index=neighbour_res, columns='res', values='psi_diff_adj')

    sc = ax.imshow(grouped, cmap='bwr')

    ax.set_xticks(range(len(grouped.columns)))
    ax.set_xticklabels(grouped.columns, fontsize=9)
    ax.set_yticks(range(len(grouped.index)))
    ax.set_yticklabels(grouped.index, fontsize=9)
    ax.set_xlabel('Current Residue')
    ax.set_ylabel(f'{neighbour_res} Residue')
    ax.set_title(f'Mean Absolute Error of ψ Prediction Based on Residue in Relation to the {neighbour_res} Residue', fontsize=14)
    plt.colorbar(sc)
    ax.set_aspect(1)

plt.savefig(f'Psi_sliding_window_residues.png')