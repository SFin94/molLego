import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import molLego as ml


if __name__ == '__main__':

    '''Script that automatically plots an existing csv file containing reaction paths or processes a reaction from the raw log files in a system .conf file and plots the reaction profile.

        The input file can be ana existing .csv or a .conf file.

        The .conf file should be formatted as:
            molKey molFile nextReactionStep

    '''

    usage = "usage: %(prog)s [inputFile(s)] [args]"
    parser = argparse.ArgumentParser(usage=usage)

    # Currently one input file only
    parser.add_argument("inputFiles", type=str, help="The .conf file with the reaction system and log files in or a .csv file of reaction data")
    parser.add_argument("-s", "--save", dest="save", type=str, default='', help="Name of csv file and plot to save, appended to _rSteps.csv, _rProfile.csv and _rPrfofile.png")
    parser.add_argument("-t", "--tparams", dest="trackParamFile", type=str, default=None, help="Name of text file containing any additional tracked parameter")
    parser.add_argument("-c", "--colour", dest="plotColour", nargs='*', default=None, help="List of colour RGB codes (starting with '#' for plotting the reaction profile in")
    parser.add_argument("-z", "--zero", "--min", dest="min", type=str, default=None, help="The reaction point (identifier in csv file) for the reaction steps to be calculated relative to")
    args = parser.parse_args()

    # Unpack inputFiles and see if csv or not
    inputFile = args.inputFiles

    # Parse in csv file of scan results
    if inputFile.split('.')[-1] == 'csv':
        reactionProfileData = pd.read_csv(inputFile, index_col=0)

    else:

        # Reads in reaction conf file and creates a molecule object for each reaction step
        reacStepNames, reacSteps = ml.constructMols(inputFile, type='thermal')
        if args.trackParamFile != None:
            parameters = ml.parseTrackedParams(args.trackParamFile)
            for rStep in reacSteps:
                if rStep.atomCoords is not None:
                    rStep.setParameters(parameters)

        # Creates dataframe of all reaction steps (global relatives and no repeats)
        reactionStepsData = ml.moleculesToDataFrame(reacSteps, molNames=reacStepNames, save=args.save + '_rSteps')

        # Calculate connectivities - could be done in class?
        paths, neighbourList = ml.constructReactionPath(inputFile, reacStepNames)
        # Then want to plot the profile
        reactionProfile = ml.initReactionProfile(reacStepNames, reacSteps, paths)

        # Create reaction profile data frame
        reactionProfileData = ml.reacProfileToDataFrame(reactionProfile, save=args.save + '_rProfile', min=args.min)

    fig, ax = ml.plotReactionProfile(reactionProfileData, save=args.save, colour=args.plotColour)
    plt.show()


