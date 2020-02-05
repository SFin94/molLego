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
    parser.add_argument("inputFiles", type=str, help="The resulting .log files with the scan in")
    parser.add_argument("-s", "--save", dest="save", type=str, help="Name of file to save plot too (minus .png extension)")
    parser.add_argument("-t", "--tparams", dest="trackParamFile", type=str, default=None, help="Name of text file containing any additional tracked parameter")
    args = parser.parse_args()

    # Unpack inputFiles and see if csv or not
    inputFile = args.inputFiles

    # Parse in csv file of scan results
    if inputFile.split('.')[-1] == 'csv':
        reactionProfileData = pd.read_csv(inputFile, index_col=0)

    else:

        # Reads in reaction conf file and creates a molecule object for each reaction step
        reacStepNames, reacStepFiles, reacSteps = ml.constructMols(inputFile, type='thermal')
        if args.trackParamFile != None:
            parameters = ml.parseTrackedParams(args.trackParamFile)
            for rStep in reacSteps:
                if rStep.atomCoords is not None:
                    rStep.setParameters(parameters)

        # Creates dataframe of all reaction steps (global relatives and no repeats)
        reactionStepsData = ml.moleculesToDataFrame(reacSteps, molNames=reacStepNames, save='reactionSteps')

        # Calculate connectivities - could be done in class?
        paths, neighbourList = ml.constructReactionPath(inputFile, reacStepNames)
        print(paths)
        # Then want to plot the profile
        reactionProfile = ml.initReactionProfile(reacStepNames, reacSteps, paths)

        # Create reaction profile data frame
        reactionProfileData = ml.reacProfileToDataFrame(reactionProfile, save='reactionProfile')

    fig, ax = ml.plotReactionProfile(reactionProfileData)
    plt.show()


