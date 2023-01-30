import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as mtick

# Load data from csv file
df = pd.read_csv("../Outputs/Output.csv", header=0)

# Set conversion factor for plotting
cm = 1/2.54

# Group data by residue and count the number of rows
residues = df.groupby('res')['index'].count().reset_index()

# Create a figure with 20 subplots
fig, axes = plt.subplots(ncols=1, nrows=20, figsize=(32*cm,480*cm))

# Rounding off the adjusted phi difference
df['phi_diff_adj_rounded'] = df['phi_diff_adj'].apply(lambda x: math.floor(x))

# Loop through each subplot
for i, ax in enumerate(axes.flatten()):

    # Get the number of rows for the current residue
    number_of_rows = residues.iloc[i, 1]
    current_residue = residues.iloc[i, 0]

    # Group data by rounded adjusted phi difference
    grouping_phi = df[df.res == current_residue].groupby('phi_diff_adj_rounded')['res'].count().reset_index()

    # Normalize the count to percentage
    grouping_phi['res'] = grouping_phi['res'].apply(lambda x: x/number_of_rows * 100)

    # Get the maximum value of the distribution
    max = grouping_phi['res'].max()

    # Plot the distribution of the phi errors
    ax.plot(grouping_phi['phi_diff_adj_rounded'], grouping_phi['res'], label="ϕ errors")

    # Add legend and set axis limits
    ax.legend(loc="upper right")
    ax.set_xlim(0,180)
    ax.set_ylim(0,max)

    # Set aspect ratio for the plot
    ax.set_aspect(180/max, adjustable='box')

    # Set x and y axis labels
    ax.set_xlabel("Error [deg]")
    ax.set_ylabel("Distribution")

    # Format the y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# Save the figure
plt.savefig(f'Distribution_per_residue.png')
