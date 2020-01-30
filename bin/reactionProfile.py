import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import molLego as ml

# Code that will have a molecule as each step in a reaction profile
# Will need the additional attributes of neighbours and reaction coordinate

# System file parser
# Now has added lines of:
# molKey molFile directedEdges
# Where directed edges are the forward connections [csv format]
# Need a product or reactant key/flag?

# Want a molFromDF

# Rectants and products - would have to be added

if __name__ == '__main__':


    '''Parse in the input log files of the scan calculations and any additional input file containing
        tracked parameters.
    '''

    usage = "usage: %(prog)s [inputFile(s)] [args]"
    parser = argparse.ArgumentParser(usage=usage)

    # Currently one input file only
    parser.add_argument("inputFiles", type=str, help="The resulting .log files with the scan in")
    parser.add_argument("-s", "--save", dest="save", type=str, help="Name of file to save plot too (minus .png extension")
    args = parser.parse_args()

    # Unpack inputFiles and see if csv or not
    inputFile = args.inputFiles[:]

    # Parse in csv file of scan results
    if inputFile.split('.')[-1] == 'csv':
        reactionProfile = pd.read_csv(args.inputFiles[0])

    else:

        # Reads in reaction conf file and creates a molecule object for each reaction step
        reacStepNames, reacStepFiles, reacSteps = ml.constructMols(inputFile, type='thermal')

        # Creates dataframe of all reaction steps (global relatives and no repeats)
        reactionStepsData = ml.moleculesToDataFrame(reacSteps, molNames=reacStepNames, save='reactionSteps')

        # Calculate connectivities - could be done in class?
        paths, neighbourList = ml.constructReactionPath(inputFile, reacStepNames)

        # Then want to plot the profile
        reactionProfile = ml.initReactionProfile(reacStepNames, reacSteps, paths)

        for rP in reactionProfile:
            for reacStep in rP.reacSteps:
                print(reacStep.g)

        # Create reaction profile data frame
        reactionProfileData = ml.reacProfileToDataFrame(reactionProfile)




