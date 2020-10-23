import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import molLego as ml


if __name__ == '__main__':
    """
    Automates constructing and plotting a reaction profile from a .conf or existing rprofile .csv.

    If the input is a reaction .conf file, this should be formatted as:
            mol_key mol_file next_reaction_step
    Where: 
        mol_key: an identifier for the current reaction step.
        mol_file: the corresponding log file.
        next_reaction_step: the mol_key for the connecting following reaction step(s) [Left empty for final step].

        E.g. for the reaction path: A + B --> C --> D + E
            R a.log,b.log Int
            Int c.log P
            P d.log,e.log
    
    """
    usage = "usage: %(prog)s [input_file(s)] [args]"
    parser = argparse.ArgumentParser(usage=usage)

    # Currently one input file only
    parser.add_argument("input_file", type=str, help="Either reaction .conf file with the reaction system or an existing reaction profile .csv file.")
    parser.add_argument("-t", "--tparams", dest="track_param_file", type=str, default=None, help="Text file containing any additional tracked parameters.")
    parser.add_argument("-c", "--colour", dest="plot_colour", nargs='*', default=None, help="List of colour RGB codes (starting with '#' for plotting the reaction profile. Number of colours should match the number of reaction paths.")
    parser.add_argument("-z", "--zero", "--min", dest="min", nargs='*', default=None, help="The identifier of the reaction stp toe calculate and plot quantities relative to.")
    
    # Unpack args and input file.
    args = parser.parse_args()
    input_file = args.input_file   
    save = input_file.split('.')[0]

    # Process input file based on type.
    if input_file.split('.')[-1] == 'csv':
        reaction_profile_data = pd.read_csv(input_file, index_col=0)
    else:
        # Create molecule objects for each reaction step in reaction conf.
        reac_step_names, reac_steps = ml.construct_mols(input_file)

        # Calculate any tracked parameters.
        if args.track_param_file != None:
            parameters = ml.parse_tracked_params(args.track_param_file)
            for step in reac_steps:
                if step.geom is not None:
                    step.set_parameters(parameters)

        # Create dataframe of all reaction steps (global relatives and no repeats).
        reaction_steps_data = ml.mols_to_dataframe(reac_steps, mol_names=reac_step_names, 
                                                    save=save + '_rsteps')

        # Calculate the reaction paths.
        reaction_paths = ml.construct_reaction_path(input_file)
        # Initilise ReactionPath objects for each of the reaction pathways.
        reaction_profile_data = ml.construct_reaction_profile(reaction_steps_data, reaction_paths, 
                                                                path_min=args.min, save=save + '_rprofile')

    # Plot reaction profile.
    fig, ax = ml.plot_reaction_profile(reaction_profile_data, save=save, colour=args.plot_colour)
    plt.show()


