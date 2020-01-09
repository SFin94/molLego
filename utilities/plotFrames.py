import numpy as np
import sys
import os
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
    fig, ax = plt.subplots(figsize=(7,6))

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
        plt.savefig(save + '.png')

    return fig, ax


def plotParamE(moleculeData, paramCol, energyCol='Relative E SCF', save=None, colour=None, scan=False, optimised=None):

    '''
    Function which plots molecules/conformers against the relative energy

    Parameters:
     moleculeData: pandas DataFrame - Containing conformer names/keys and energies
     paramCol: str - header for parameter column in dataframe
     energyCol: str - header for relative energy column in dataframe [default: 'Relative E']
     save: str - name of image to save plot too (minus .png extension) [deafult: None type]
     colour: colour (matplotlib) - colour to plot the conformers in [default: None type]. If default then a cubehelix colour is used.
     scan: bool -

    Returns:
     fig, ax - :matplotlib:fig, :matplotlib:ax objects for the plot

    '''

    fig, ax = plotSetup()

    if colour == None:
        colour = sns.cubehelix_palette(8, start=.5, rot=-.4, dark=0, light=0.5)

    # Change to make - set colour for unoptimised and set optimised as another class feature
    # Set colour list so that unoptimised points are coloured differently
    # If optimised/non-optimised information is known then set plot colours to show unopt points
    if optimised != None:
        colList = []
        colours = ['#E71E47', '#0C739C']
#        colours = ['#E71E47', colour[5]]
        [colList.append(colours[opt]) for opt in optimised]
    else:
#        colList = [colour[5]]*len(list(moleculeData.index))
        colList = ['#0C739C']*len(list(moleculeData.index))

    # Plot points and connecting lines
    ax.scatter(moleculeData[paramCol], moleculeData[energyCol], color=colList, marker='o', s=50, alpha=0.6)
    ax.plot(moleculeData[paramCol], moleculeData[energyCol], marker=None, alpha=0.3, color='#0C739C')

    # Set x and y labels
    ax.set_xlabel(paramCol, fontsize=13)
    ax.set_ylabel('$\Delta$E (kJmol$^{-1}$)', fontsize=13)

    # Set legend to show unopt vs. opt points
    ax.legend(handles=[mlin.Line2D([], [], color='#E71E47', label='Unoptimised', marker='o', alpha=0.6, linestyle=' '), mlin.Line2D([], [], color='#0C739C', label='Optimised', marker='o', alpha=0.6, linestyle=' ')], frameon=False, handletextpad=0.1)

    if save != None:
        plt.savefig(save[0] + '.png')

    return fig, ax
