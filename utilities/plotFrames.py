import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlin
from scipy.interpolate import griddata
import seaborn as sns


'''Script with some general plotting functions'''

def plotSetup():

    '''
    Function that sets some general settings for all plots

    Returns:
     fig, ax - :matplotlib:fig, :matplotlib:ax objects for the plot
    '''

    # Set font parameters and colours
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    # Set figure and plot param(s) vs energy
    fig, ax = plt.subplots(figsize=(12,9))

    # Remove plot frame lines
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return fig, ax


def plotMolsE(moleculeData, energyCol='Relative E', save=None, colour=None, labels=None):

    '''
    Function which plots molecules/conformers against the relative energy

    Parameters:
     moleculeData: pandas DataFrame - Containing conformer names/keys and energies
     energyCol: str - header for relative energy column in dataframe [default: 'Relative E']
     save: str - name of image to save plot too (minus .png extension) [deafult: None type]
     colour: colour (matplotlib) - colour to plot the conformers in [default: None type]. If default then a cubehelix colour is used.

    Returns:
     fig, ax - :matplotlib:fig, :matplotlib:ax objects for the plot

    '''

    fig, ax = plotSetup()

    if colour == None:
        colour = sns.cubehelix_palette(8, start=.5, rot=-.4, dark=0, light=0.5)

    # Plot conformer vx. relative energy
    ax.scatter(list(moleculeData.index), moleculeData[energyCol], marker='o', alpha=0.8, color=colour[5], s=70)

    if labels == None:
        labels = moleculeData.index

    # Set x and y labels and ticks
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Relative Energy (kJmol$^{-1}$)')
    ax.set_xlabel('Molecule')

    if save != None:
        plt.savefig(save + '.png', dpi=600)

    return fig, ax


def plotParamE(moleculeData, paramCol, energyCol='Relative E SCF', save=None, colour=None, scan=False):

    '''
    Function which plots molecules/conformers against the relative energy

    Parameters:
     moleculeData: pandas DataFrame - Containing conformer names/keys and energies
     paramCol: str - header for parameter column in dataframe
     energyCol: str - header for relative energy column in dataframe [default: 'Relative E']
     save: str - name of image to save plot too (minus .png extension) [deafult: None type]
     colour: colour (matplotlib) - colour to plot the conformers in [default: None type; if default then a cubehelix colour is used].
     scan: bool - flag of whether a scan is being plotted, if true then links the scatterpoints with a line

    Returns:
     fig, ax - :matplotlib:fig, :matplotlib:ax objects for the plot

    '''

    fig, ax = plotSetup()

    # Set colours for plotting
    if 'Optimised' in moleculeData.columns.values:
        if colour == None:
            colour = [sns.cubehelix_palette(8, start=2.1, dark=0, light=0.5)[5], sns.cubehelix_palette(8, dark=0, light=0.5)[5]]
            colour = [sns.cubehelix_palette(8)[5], sns.cubehelix_palette(8, rot=-.4)[5]]
        colList = []
        [colList.append(colour[opt]) for opt in moleculeData['Optimised']]
    else:
        colList = [colour]*len(list(moleculeData.index))

    # Plot points and connecting lines if scan
    ax.scatter(moleculeData[paramCol], moleculeData[energyCol], color=colList, marker='o', s=50, alpha=0.8)
    if scan == True:
        ax.plot(moleculeData[paramCol], moleculeData[energyCol], marker=None, alpha=0.4, color=colour[1])

    # Set x and y labels
    ax.set_xlabel(paramCol, fontsize=13)
    ax.set_ylabel('$\Delta$E (kJmol$^{-1}$)', fontsize=13)

    # Set legend to show unopt vs. opt points
    if 'Optimised' in moleculeData.columns.values:
        ax.legend(handles=[mlin.Line2D([], [], color=colour[0], label='Unoptimised', marker='o', alpha=0.6, linestyle=' '), mlin.Line2D([], [], color=colour[1], label='Optimised', marker='o', alpha=0.6, linestyle=' ')], frameon=False, handletextpad=0.1)

    if save != None:
        plt.savefig(save + '.png', dpi=600)

    return fig, ax


def plotPES(moleculeData, paramOneCol, paramTwoCol, energyCol='Relative E SCF', save=None, colour=None):

    '''
    Function which plots a 2D PES for two parameters

     Parameters:
      moleculeData: pandas DataFrame - Containing conformer names/keys and energies
      paramOneCol: str - header for first parameter column in dataframe
      paramTwoCol: str - header for second parameter column in dataframe
      energyCol: str - header for relative energy column in dataframe [default: 'Relative E']
      save: str - name of image to save plot too (minus .png extension) [deafult: None type]
      colour: colour (matplotlib cmap) - colour map to plot the PES [default: None type; if default then a cubehelix colour map is used].

     Returns:
      fig, ax - :matplotlib:fig, :matplotlib:ax objects for the plot

     '''

    fig, ax = plotSetup()

    # Filter out any unoptimised points
    moleculeData = moleculeData[moleculeData.Optimised]

    # Set linearly spaced parameter values and define grid between them
    paramOneRange = np.linspace(moleculeData[paramOneCol].min(), moleculeData[paramOneCol].max(), 100)
    paramTwoRange = np.linspace(moleculeData[paramTwoCol].min(), moleculeData[paramTwoCol].max(), 100)
    paramOneGrid, paramTwoGrid = np.meshgrid(paramOneRange, paramTwoRange)

    # Interpolate the energy data on to the grid points for plotting
    interpE = griddata((moleculeData[paramOneCol].values, moleculeData[paramTwoCol].values), moleculeData[energyCol], (paramOneGrid, paramTwoGrid), method='cubic')

    # Set cmap if none provided
    if colour == None:
        colour = sns.cubehelix_palette(light=1, dark=0, as_cmap=True)

    # Plot filled contour and add colour bar
    c = ax.contourf(paramOneRange, paramTwoRange, interpE, 20, cmap=colour)
    fig.subplots_adjust(right=0.8)
    cb = fig.colorbar(c)
    cb.set_label('Relative Energy (kJmol$^{-1}$)')

    # Plot scatter points on (all; TS3b_sp; TS3b_gp_guess)
#    ax.plot(scanResults[paramOne], scanResults[paramTwo], marker='o', alpha=0.4, linestyle=' ', color='#071047')
#    ax.plot(2.112312163834929, 1.284186052611147, marker='o', alpha=0.7, color='#071047')

    # Set x and y labels
    ax.set_xlabel(paramOneCol, fontsize=13)
    ax.set_ylabel(paramTwoCol, fontsize=13)

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax


