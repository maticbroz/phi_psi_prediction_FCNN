import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_phi_psi_angles(df):
    phi = df['phi']
    phi_pred_adj = df['phi_pred_adj']
    psi = df['psi']
    psi_pred_adj = df['psi_pred_adj']
    
    # Linear regression of phi and phi_pred_adj
    phi_regression = stats.linregress(phi, phi_pred_adj)
    print("Linear Regression Results for Phi:", phi_regression)
    
    # Linear regression of psi and psi_pred_adj
    psi_regression = stats.linregress(psi, psi_pred_adj)
    print("Linear Regression Results for Psi:", psi_regression)
    
    # Plotting predicted Phi and Psi angles
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(phi_pred_adj, psi_pred_adj, 'b.', markersize=0.2)
    ax.set_title("Predicted Phi vs Predicted Psi Angles")
    ax.set_xlabel('Predicted [deg]')
    ax.set_ylabel('Predicted [deg]')
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

# Read the csv file and plot the Phi and Psi angles
df = pd.read_csv("../Outputs/Output.csv", header=0)
plot_phi_psi_angles(df)
