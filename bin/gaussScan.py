import argparse
import matplotlib.pyplot as plt
import pandas as pd
import molLego as ml


if __name__ == '__main__':

    '''Parse in the input log files of the scan calculations and any additional input file containing
        tracked parameters.
    '''

    usage = "usage: %(prog)s [inputFile(s)] [args]"
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument("inputFiles", nargs='*', type=str, help="The resulting .log files with the scan in")
    parser.add_argument("-p", "--sparam", dest="scanParam", type=str)
    parser.add_argument("-t", "--tparam", dest="trackParamFile", type=str, default=None)
    parser.add_argument("--noplot", dest="plot", action='store_false')
    parser.add_argument("-s", "--save", dest="save")
    args = parser.parse_args()

    # Unpack inputFiles and see if csv or not
    inputFiles = args.inputFiles[:]


    # Parse in csv file of scan results
    if inputFiles[0].split('.')[-1] == 'csv':
        scanResults = pd.read_csv(args.inputFiles[0])
        try:
            scanParameter = args.scanParam[0]
        except:
            print("No scan parameter given, provide the header for the scan parameter column from csv as: -p [header]")

    # Create scan molecules and csv for scan log files
    else:
        scanFiles, scanMolecules = ml.initScan(*inputFiles, trackedParams=args.trackParamFile)
        scanInfo = ml.pullScanInfo(inputFiles[0])
        scanParameter = scanInfo['paramKey']
        scanResults = ml.moleculesToDataFrame(scanFiles, scanMolecules, save='ps'+scanParameter)

    # Plot results if wanted (sort results by parameter value first)
    if args.plot == True:
        scanResults = scanResults.sort_values(scanParameter)
        fig, ax = ml.plotParamE(scanResults, paramCol=scanParameter, save=args.save)
        plt.show()


