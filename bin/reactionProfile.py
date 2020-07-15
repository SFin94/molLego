import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import molLego as ml


if __name__ == '__main__':

    '''Script that automatically plots an existing csv file containing reaction paths or processes a reaction from the raw log files in a system .conf file and plots the reaction profile.

        The input file can be ana existing .csv or a .conf file.

        The .conf file should be formatted as:
            mol_key mol_file next_reaction_step
        Where mol_key is an identifier for the current reaction step, mol_file is the corresponding log file and next_reaction_step is the mol_key for the connecting following reaction step(s) [None for products]

        E.g. for the reaction path: A --> B --> C
            A   a.log   B
            B   b.log   C
            C   c.log
    '''

    usage = "usage: %(prog)s [input_file(s)] [args]"
    parser = argparse.ArgumentParser(usage=usage)

    # Currently one input file only
    parser.add_argument("input_file", type=str, help="The .conf file with the reaction system and log files in or a .csv file of reaction data")
    parser.add_argument("-s", "--save", dest="save", type=str, default='', help="Name of csv file and plot to save, appended to _rsteps.csv, _rprofile.csv and _rprofile.png")
    parser.add_argument("-t", "--tparams", dest="track_param_file", type=str, default=None, help="Name of text file containing any additional tracked parameter")
    parser.add_argument("-c", "--colour", dest="plot_colour", nargs='*', default=None, help="List of colour RGB codes (starting with '#' for plotting the reaction profile in")
    parser.add_argument("-z", "--zero", "--min", dest="min", type=str, default=None, help="The reaction point (identifier in csv file) for the reaction steps to be calculated relative to")
    
    # Unpack args and input file
    args = parser.parse_args()
    input_file = args.input_file

    # Read in if csv file or process reaction pathways from .conf file
    if input_file.split('.')[-1] == 'csv':
        reaction_profile_data = pd.read_csv(input_file, index_col=0)

    else:

        # Reads in reaction conf file and creates a molecule object for each reaction step
        reac_step_names, reac_steps = ml.construct_mols(input_file)
        if args.track_param_file != None:
            parameters = ml.parse_tracked_params(args.track_param_file)
            for step in reac_steps:
                if step.geom is not None:
                    step.set_parameters(parameters)

        # Creates dataframe of all reaction steps (global relatives and no repeats)
        reaction_steps_data = ml.mols_to_dataframe(reac_steps, mol_names=reac_step_names, save=args.save + '_rsteps')

        # Calculate the reaction paths 
        reaction_paths = ml.construct_reaction_path(input_file, reac_step_names)
        # Initilise ReactionPath objects for each of the reaction pathways
        reaction_profile = ml.init_reaction_profile(reac_step_names, reac_steps, reaction_paths)

        # Create reaction profile data frame
        reaction_profile_data = ml.reaction_profile_to_dataframe(reaction_profile, save=args.save + '_rprofile', min=args.min)

    # Plot reaction profile
    fig, ax = ml.plot_reaction_profile(reaction_profile_data, save=args.save, colour=args.plot_colour)
    plt.show()


