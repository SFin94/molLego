import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import molLego as ml


if __name__ == '__main__':

    '''Script that automatically plots molecules/conformers against energy values from an existing csv file or processes the data from the raw log files in a system .conf file.

        The input file can be an existing .csv or .conf file.

        The .conf file should be formatted as:
            molKey molFile
        Where the molKey will be the identifier of the molecule, if certain conformers are to be plotted they should be passed as a list of the relevant molKeys.

        The tracked parameters file should be formatted as:
            parameterKey atomInd1 atomInd2 [atomInd3 [atomInd4]]
        Where 2, 3 or 4 atom indexes (int) can be provided for bonds, angle or dihedrals. The indexes match those in the gaussian log file. The parameterKey will be both the dict key for the indexes and the heading in the results file for the parameter values.

    '''

    usage = "usage: %(prog)s [inputFile(s)] [args]"
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument("inputFiles", type=str, help="The .conf or .csv file with the molecules/conformers to be plotted")
    parser.add_argument("-s", "--save", dest="save", type=str, default='', help="Name of csv file and plot to save, appended to .csv and .png extensions")
    parser.add_argument("-t", "--tparams", dest="trackParamFile", type=str, default=None, help="Name of text file containing any additional tracked parameter")
    parser.add_argument("-c", "--confs", dest="confsToPlot", type=list, default=None, help="List of the mol/conformer keys of the moelcules to be plotted from the input")
    parser.add_argument("--type", dest="molType", type=str, default='molecule', choices=['molecule', 'thermal'], help="Type of molecule, can be simple ('molecule') or thermal if frequency calculation has been performed for thermodynamic data")
    args = parser.parse_args()

    # Open system file, create molecule objects and dataframe or parse existing data frame
    if args.inputFiles.split('.')[1] == 'conf':
        molkeys, molecules = ml.constructMols(sys.argv[1], type=args.molType)

        # Optional - add in tracked geometric parameters if provided
        if args.trackParamFile != None:
            parameters = ml.parseTrackedParams(args.trackParamFile)
            for mol in molecules:
                mol.setParameters(parameters)

        csvFile = sys.argv[1].split('.')[0]
        molData = ml.moleculesToDataFrame(molecules, molNames=molkeys, save=args.save)

    # If input is csv then process into dataframe
    elif args.inputFiles.split('.')[1] == 'csv':
        molData = pd.read_csv(sys.argv[1], index_col=0)

    # Process data to calculate relative values and plot conformers vs. energy (raw SCF energy if no thermal quantities available)
    if any(val == 'G' for val in molData.columns.values):
        molData = ml.calcRelative(molData, molsToPlot=args.confsToPlot, quantities=['E', 'G', 'H'])
        fig, ax = ml.plotMolsAll(molData, save=args.save)

    else:
        molData = ml.calcRelative(molData, molsToPlot=args.confsToPlot)
        fig, ax = ml.plotMolsE(molData, save=args.save, energyCol='Relative E SCF')

    plt.show()
