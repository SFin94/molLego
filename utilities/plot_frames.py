"""Module containing general plotting routines for molecules."""

import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.axes as axes
import matplotlib.lines as mlin
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata

def plot_setup(figsize_x=8, figsize_y=6, fig=None, ax=None):
    """
    Initialise plot with general settings.

    Parameters
    ----------
    figsize_x : `int`
        x dimension of plot [default: 8]
    figsize_y : `int`
        y dimension of plot [default: 6]
    fig : :matplotlib:`fig`
        If other type of plot is called first [default: None]
    ax : :matplotlib:`axes`
        If other type of plot is called first [default: None]

    Returns
    -------
    fig, ax: :matplotlib:`fig`, :matplotlib:`axes` for the plot.
    
    """
    # Set font parameters and colours
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    colour_grey = '#3E3E3E'
    plt.rcParams.update({'text.color': colour_grey, 
                         'axes.labelcolor': colour_grey, 
                         'xtick.color': colour_grey, 
                         'ytick.color': colour_grey})

    # Initiaise figure.
    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize=(figsize_x,figsize_y))

    # Remove lines from plot frame.
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=12)

    return fig, ax


def radial_plot_setup(figsize_x=6, figsize_y=6, fig=None, ax=None):
    """
    Initialise radial plot with general settings.

    Parameters
    ----------
    figsize_x : `int`
        x dimension of plot [default: 6]
    figsize_y : `int`
        y dimension of plot [default: 6]
    fig : :matplotlib:`fig`
        If other type of plot is called first [default: None]
    ax : :matplotlib:`axes`
        If other type of plot is called first [default: None]

    Returns
    -------
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.
    
    """
    # Set font parameters and colours
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    colour_grey = '#3E3E3E'
    plt.rcParams.update({'text.color': colour_grey, 
                         'axes.labelcolor': colour_grey, 
                         'xtick.color': colour_grey, 
                         'ytick.color': colour_grey})

    # Set figure and plot param(s) vs energy
    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize=(figsize_x,figsize_y), 
                               subplot_kw=dict(projection='polar'))

    # ax.spines["circle"].set_visible(False)
    ax.tick_params(labelsize=12)

    return fig, ax

def plot_mols_E(mol_data, energy_col=['Relative E'], save=None, 
                colour=None, mol_labels=None, line=False):
    """
    Plot molecules/conformers against relative energy.

    Parameters
    ----------
    mol_data : :pandas:`DataFrame`
        DataFrame containing molecule results.
    energy_col : str`
        Column header corresponding to quantity to plot molecules by.
        [default: Relative E]
    save : `str`
        File name to save figure as (minus .png extension). 
        [default: None; no figure is saved]
    colour : `list of str`
        Colour list to plot molecules by.
        [default: None type; uses cubehelix colours].
    mol_labels : `list of str`
        Molecule identifiers if different to DatFrame index.
        [default: None type, uses DataFrame index]
    line : `bool`
        If True, connects scatterpoints by lines.
        
    Returns
    -------
    fig, ax : :matplotlib:fig, :matplotlib:ax for the plot

    """
    fig, ax = plot_setup(figsize_x=8, figsize_y=7)

    # Plot conformer vs. relative energy
    if type(energy_col) != list:
        energy_col = [energy_col]
    if colour == None:
        colour = sns.cubehelix_palette(len(energy_col), start=.5, 
                                     rot=-.4, dark=0, light=0.5)

    # Plot conformers for each quantity
    for col_ind, col in enumerate(energy_col):
        ax.scatter(list(mol_data.index), mol_data[col], 
                        marker='o', alpha=0.8, color=[col_ind], 
                        label=col, s=70)
        if line == True:
            ax.plot(list(mol_data.index), mol_data[col], alpha=0.3, 
                         color=colour[col_ind], ls='--')

    if mol_labels == None:
        mol_labels = mol_data.index

    # Set x and y labels and ticks
    ax.set_xticklabels(mol_labels, rotation=15)
    ax.set_ylabel('Relative Energy (kJmol$^{-1}$)', fontsize=13)
    ax.set_xlabel('Molecule', fontsize=13)
    plt.legend(fontsize=13)

    if save != None:
        plt.savefig(save + '.png', dpi=600)

    return fig, ax


def plot_mols_thermo(mol_data, save=None, mol_labels=None, enthalpy=False):
    """
    Plot molecules against relative E and G (and H).

    Parameters
    ----------
    mol_data : :pandas:`DataFrame`
        DataFrame containing molecule results.
    save : `str`
        File name to save figure as (minus .png extension). 
        [deafult: None; no figure is saved]
    mol_labels : `list of str`
        Molecule identifiers if different to DatFrame index.
        [default: None type, uses DataFrame index]
    enthalpy :
        If True also plots enthalpy values.
        [default: False]

    Returns
    -------
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

    """
    fig, ax = plot_setup()

    # Set colours for G and E
    e_colour = '#245F6B'
    g_colour = '#D17968'

    # Plot products vs. energy
    ax.scatter(mol_data.index, mol_data['Relative E'], 
               color=e_colour, s=70, label='$\Delta$E')
    ax.scatter(mol_data.index, mol_data['Relative G'], 
               color=g_colour, s=70, label='$\Delta$G')

    # Plot enthalpy (H) too if flag is True
    if enthalpy == True:
        h_colour = '#175443'
        ax.scatter(mol_data.index, mol_data['Relative H'], 
                   color=h_colour, s=70, label='$\Delta$H')

    # Set labels and axis settings
    if mol_labels == None:
        mol_labels = list(mol_data.index)

    ax.tick_params(labelsize=10)
    ax.set_xticklabels(mol_labels, rotation=15, fontsize=11)
    ax.set_ylabel('Relative E/G (kJmol$^{-1}$)', fontsize=11)
    ax.set_xlabel('Molecule', fontsize=11)
    ax.legend(frameon=False, loc=1)

    if save != None:
        plt.savefig(save + '.png', dpi=600)

    return fig, ax


def plot_param_E(mol_data, param_col, energy_col='Relative E', 
                 save=None, colour=None, scan=False):
    """
    Plot relative energies (or other specified quantity) of molecules.

    Parameters
    ----------
    mol_data : :pandas:`DataFrame`
        DataFrame containing molecule results.
    param_col : `str`
        Column header corresponding to geometric parameter to plot.
    energy_col : `str`
        Column header corresponding to quantity to plot molecules by.
        [default: Relative E]
    save : `str`
        File name to save figure as (minus .png extension). 
        [deafult: None; no figure saved]
     colour : `str`
        A colour specified by a recognised matplotlib format.
        [default: None type; uses cubehelix colours].
     scan : `bool`
        If True, connects scatterpoints by lines.
        [default: False]

    Returns
    -------
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

    """
    fig, ax = plot_setup()

    # Set colours for plotting if not provided
    if colour == None:
        colour = ['#D17968', '#12304e']
    elif len(colour[0]) == 1:
        colour = [colour]

    # Set colours by molecule opt value.
    if 'Optimised' in mol_data.columns.values:
        colour_list = []
        [colour_list.append(colour[opt]) for opt in mol_data['Optimised']]
    elif len(colour) == len(list(mol_data.index)):
        colour_list = colour
    else:
        colour_list = [colour[0]]*len(list(mol_data.index))

    # Plot points and connecting lines if scan
    ax.scatter(mol_data[param_col], mol_data[energy_col], 
               color=colour_list, marker='o', s=70, alpha=0.8)
    if scan == True:
        ax.plot(mol_data[param_col], mol_data[energy_col], 
                marker=None, alpha=0.4, color=colour[1])

    # Set x and y labels
    ax.set_xlabel(param_col, fontsize=11)
    ax.set_ylabel('$\Delta$E (kJmol$^{-1}$)', fontsize=11)

    # Set legend to show unopt vs. opt points
    if 'Optimised' in mol_data.columns.values:
        ax.legend(handles=[mlin.Line2D([], [], color=colour[0], 
                  label='Unoptimised', marker='o', alpha=0.6, linestyle=' '), 
                  mlin.Line2D([], [], color=colour[1], label='Optimised', 
                  marker='o', alpha=0.6, linestyle=' ')], frameon=False, 
                  handletextpad=0.1, fontsize=10)

    if save != None:
        plt.savefig(save + '.png', dpi=600)

    return fig, ax


def plot_PES(mol_data, param_cols, energy_col='Relative E', 
             save=None, colour=None, opt_filter=True):
    """
    Plot 2D PES for two geometric parameters.

    Parameters
    ----------
    mol_data : :pandas:`DataFrame`
        DataFrame containing molecule results.
    param_cols : `list of str`
        Column headers corresponding to the two geometric parameter to plot.
    energy_col : ``str``
        Column header corresponding to quantity to plot molecules by.
        [default: Relative E]
    save : `str`
        File name to save figure as (minus .png extension). 
        [deafult: None; no figure is saved]
    colour : :matplotlib:`cmap`
        Colour map to plot PES. 
        [default: None type; uses cubehelix colour map].
    opt_filter : `bool`
        If True then removes unoptimised data points.
        [default: True]

    Returns
    -------
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

    """
    fig, ax = plot_setup(figsize_x=7.5, figsize_y=6)

    # Filter out any unoptimised points if optimised present
    opt_col = ('Optimised' in mol_data.columns.values)
    if all([opt_filter, opt_col]):
        mol_data = mol_data[mol_data.Optimised]

    # Set linearly spaced parameter values and define grid between them
    param_one_range = np.linspace(mol_data[param_cols[0]].min(), 
                                  mol_data[param_cols[0]].max(), 100)
    param_two_range = np.linspace(mol_data[param_cols[1]].min(), 
                                  mol_data[param_cols[1]].max(), 100)
    param_one_grid, param_two_grid = np.meshgrid(param_one_range, 
                                                 param_two_range)

    # Interpolate the energy data on to the grid points for plotting
    interp_E = griddata((mol_data[param_cols[0]].values, 
                         mol_data[param_cols[1]].values), 
                         mol_data[energy_col], 
                         (param_one_grid, param_two_grid))

    # Set cmap if none provided
    if colour == None:
        colour = sns.cubehelix_palette(dark=0, as_cmap=True)

    # Plot filled contour and add colour bar
    c = ax.contourf(param_one_range, param_two_range, interp_E, 
                    20, cmap=colour, vmax=150)
    fig.subplots_adjust(right=0.8)
    cb = fig.colorbar(c)
    cb.set_label('$\Delta$E (kJmol$^{-1}$)', fontsize=13)

    # Set x and y labels
    ax.set_xlabel(param_cols[0], fontsize=13)
    ax.set_ylabel(param_cols[1], fontsize=13)

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax

def plot_reaction_profile(reaction_data, quantity_col='Relative G', save=None,
                            colour=None, step_width=3000, line_buffer=0.08, 
                            label=True, fig=None, ax=None):
    """
    Plot a reaction profile.

    Parameters
    ----------
    reaction_data : :pandas:`DataFrame`
        The reaction profile dataframe to plot.
    energy_col : ``str``
        Column header of quantity to plot reaction steps by.
        [default: 'Relative G']
    save : `str`
        File name to save figure as (minus .png extension). 
        [deafult: None; no figure is saved]
    colour : :matplotlib:`cmap`
        Colour map to generate path plot colours from. 
        [default: None type; uses cubehelix colour map].
    step_width : `int`
        The marker size of the scatter hlines used to mark the reaction step.
        [default: 3000]
    line_buffer : `float`
        The buffer from centre of the hline of position
        the connecting lines will connect to.
        [default: 0.05]
    label : `bool`
        If True then plots the indexes with each step.
        If False then returns the figure without labels.
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

    Returns
    -------
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

    """
    fig, ax = plot_setup(fig=fig, ax=ax)
    paths = list(reaction_data['Reaction path'].unique())

    # Set colours if not provided.
    if colour == None:
        col_pallete = sns.color_palette("cubehelix", len(paths))
        colour = []
        for p_ind in range(len(paths)):
            colour.append(col_pallete[paths.index(p_ind)])

    # Plot lines and points for the profile.
    for p_ind, path in enumerate(paths):
        reac_path_data = reaction_data.loc[
                            reaction_data['Reaction path'] == path]
        ax.scatter(reac_path_data['Rx'], reac_path_data[quantity_col], 
                   color=colour[p_ind], marker='_', s=step_width, lw=8)

        # line_buffer and step_width can be altered to fit the profile.
        for rstep_ind in range(1, len(reac_path_data)):
            ax.plot([reac_path_data['Rx'].iloc[rstep_ind-1]+line_buffer, 
                     reac_path_data['Rx'].iloc[rstep_ind]-line_buffer], 
                     [reac_path_data[quantity_col].iloc[rstep_ind-1], 
                     reac_path_data[quantity_col].iloc[rstep_ind]],  
                     color=colour[p_ind], linestyle='--')

            # Plot labels with dataframe index and energy label.
            if label == True:
                step_label = (reac_path_data.index.values[rstep_ind] 
                              + '\n(' + str(int(reac_path_data[
                              quantity_col].iloc[rstep_ind])) + ')')
                ax.text(reac_path_data['Rx'].iloc[rstep_ind], 
                        reac_path_data[quantity_col].iloc[rstep_ind]+6, 
                        step_label, color=colour[p_ind], fontsize=11, 
                        horizontalalignment='center')
        
        # Plot labels of reactants.
        if label == True:
            reactant_label = (reac_path_data.index.values[0] 
                              + '\n(' + str(int(reac_path_data[
                              quantity_col].iloc[0])) + ')')
            ax.text(reac_path_data['Rx'].iloc[0], 
                    reac_path_data[quantity_col].iloc[0]+6, 
                    reactant_label, color=colour[p_ind], 
                    fontsize=11, horizontalalignment='center')

    # Set figure properties.
    ax.set_xlabel('R$_{x}$', fontsize=13)
    ax.set_ylabel('$\Delta$G (kJmol$^{-1}$)', fontsize=13)
    ax.set_xticks([])

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax

def normalise_parameters(mol_data, geom_params):
    """
    Update DataFrame with bond/angle/dihedrals mapped to a 0:1 range.

    Distances are normalised to [0:1] range
    Angles are mapped from [0:180] range to [0:1] range
    Dihedrals are mapped from [-180:180] range to [0:1] range

    Parameters
    ----------
    mol_data : :pandas:`DataFrame`
        DataFrame containing molecule results to be normalised.
    geom_params : `dict`
        Key is column heading and 
        value is the atom indexes of the parameter.

    Returns
    -------
    param_headings : `list of str`
        Parameter headings for the normalised parameters.

    """
    param_headings = []
    for key, value in geom_params.items():
        if len(value) == 2:
            mol_data["Norm " + key] = mol_data[key]/mol_data[key].max()
        elif len(value) == 3:
            mol_data["Norm " + key] = mol_data[key]/180.
        else:
            mol_data["Norm " + key] = (mol_data[key]%360.)/360.

        # Set parameter heading
        param_headings.append("Norm " + key)

    return param_headings

def set_mol_colours(mol_data, energy_col):
    """
    Set colours for different conformers, can represent energy values.

    Parameters
    ----------
    mol_data : :pandas:`DataFrame`
        DataFrame containing molecule results.
    energy_col : ``str``
        Column header corresponding to quantity to code colours by.
        [default: None]

    Returns
    -------
    colours : `list`
        Colour code corresponding to each conformer.
    
    """
    # Calculate normalised energy to plot colour by if given.
    if energy_col != None:
        mol_data['Norm E'] = mol_data[energy_col]/mol_data[energy_col].max()
        colmap = sns.cubehelix_palette(as_cmap=True)
        for val in mol_data['Norm E']:
            colours = [colmap(val)[:3] for val in mol_data['Norm E']]
    else:
    # Else set colours different for each conformer.
        colours = sns.cubehelix_palette(len(mol_data.index))
    return colours


def plot_mols_radar(mol_data, geom_params, save=None, 
                    colour=None, energy_col=None):
    """
    Plot molecules against multiple geometric parameters in a radial plot.

    Parameters
    ----------
    mol_data : :pandas:`DataFrame`
        DataFrame containing molecule results.
    geom_params : `dict`
        Key is column heading and value is the atom indexes of the parameter.
    save : `str`
        File name to save figure as (minus .png extension). 
        [deafult: None; no figure is saved]
    colour : `list of str`
        Colour list to plot conformers by.
        [default: None type; uses cubehelix colours].
    energy_col : ``str``
        Column header corresponding to quantity to code colours by.
        [default: None]

    Returns
    -------
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

    """
    fig, ax = radial_plot_setup()

    # Calculate angles to plot, set parameter list
    num_params = len(geom_params.keys())
    plot_angles = [n / float(num_params) * 2 * np.pi 
                   for n in range(num_params)]
    plot_angles += plot_angles[:1]

    # Normalise molecule parameters.
    param_headings = normalise_parameters(mol_data, geom_params)
    param_headings.append(param_headings[0])

    # Set colour for molecules.
    if colour == None:
        mol_data['Colour'] = set_mol_colours(mol_data, energy_col)
    else:
        mol_data['Colour'] = colour

    # Plot for each conformer
    for mol in mol_data.index:
        ax.plot(plot_angles, mol_data.loc[mol, param_headings], 
                label=mol, color=mol_data.loc[mol, 'Colour'])
        ax.fill(plot_angles, mol_data.loc[mol, param_headings], 
                color=mol_data.loc[mol, 'Colour'], alpha=0.1)

    # Set plot attributes
    ax.set_xticks(plot_angles[:-1])
    ax.set_xticklabels(list(geom_params.keys()))
    ax.set_yticks([])
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.04), ncol=3, 
              frameon=False, handletextpad=0.1, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax


def plot_mol_map(mol_data, geom_params, save=None, 
                  colour=None, energy_col=None):
    """
    Plot molecules against several geometric parameters in a linear plot.

    Parameters
    ----------
    mol_data:  :pandas:`DataFrame`
        DataFrame containing molecule results.
    geom_params : `dict`
        Key is column heading and value is the atom indexes of the parameter.
    save : `str`
        File name to save figure as (minus .png extension). 
        [deafult: None; no figure is saved]
    colour : `list of str`
        Colour list to plot conformers by.
        [default: None type; uses cubehelix colours].
    energy_col : ``str``
        Column header corresponding to quantity to code colours by.
        [default: None]

    Returns
    -------
    fig, ax : :matplotlib:`fig`, :matplotlib:`axes` for the plot.

    """
    fig, ax = plot_setup()
    num_params = len(geom_params.keys())
    plot_params = list(geom_params.keys())

    # Normalise molecule parameters.
    param_headings = normalise_parameters(mol_data, geom_params)

    # Set colour for molecules.
    if colour == None:
        mol_data['Colour'] = set_mol_colours(mol_data, energy_col)
    else:
        mol_data['Colour'] = colour

    # Plot data.
    for i, mol in enumerate(mol_data.index):
        ax.plot(range(num_params), mol_data.loc[mol, param_headings], label=mol, 
                color=mol_data.loc[mol, 'Colour'], marker='o', alpha=0.8)

    # Set x and y labels and ticks
    ax.set_xticks(range(num_params))
    ax.set_xticklabels(plot_params, rotation=20, ha='right')
    ax.set_ylim(ymin=0.0, ymax=1.0)

    # Set legend.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.04), ncol=3, 
              frameon=False, handletextpad=0.1, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save != None:
        plt.savefig(save + '.png')

    return fig, ax
