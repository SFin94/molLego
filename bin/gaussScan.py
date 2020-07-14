import argparse
import matplotlib.pyplot as plt
import pandas as pd

import molLego as ml


'''
Script to store results from gaussian relaxed scan log file(s) to a dataframe (csv) and to optionally plot them or to plot results from an existing dataframe (csv file).

Uses argparse for CL arguments
    Usage:"usage: %(prog)s [input_files] [args]"
    
Where:
    input_files: str - paths to either multiple gaussian log files (same parameter scanned in all of them); OR csv file with previous dataframe in [NB: if csv then scan parameter needs to be specified]
    args:
        '-p/--sparam': str - scan parameter (corresponding to header of parameter column in csv file); only needs to be included if input file is csv file
        '-t/--tparam': str - path to additional file containing other geometric parameters to track
        's/--save': str - name to save plot with (minus .png extension)
        '--noplot': bool [default:True] - if flag used then stores False and plot will not be plotted (e.g. if only want csv files from original log files)
'''

if __name__ == '__main__':

    '''Parse in the input log files of the scan calculations and any additional input file containing tracked parameters.
    '''

    usage = "usage: %(prog)s [input_file(s)] [args]"
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument("input_files", nargs='*', type=str, help="The resulting .log files with the scan in")
    parser.add_argument("-p", "--params", dest="scan_params", type=list, help="Scan parameters (matches dataframe column header if being parsed from csv)")
    parser.add_argument("-t", "--tparam", dest="track_param_file", type=str, default=None, help="Name of text file containing any additional tracked parameter")
    parser.add_argument("--plot", dest="plot", type=int, default=1, help="The dimensionality of the surface to be plotted, 0: no plot; 1: 1D PE profile; 2: 2D PES")
    parser.add_argument("-s", "--save", dest="save", type=str, help="Name of file to save plot too (minus .png extension")
    parser.add_argument("-o", "--filteropt", dest="opt_filter", type=bool, default=False, help="Flag of whether to filter unoptimised points from dataframe or not (if rigid, then must be set to True")
    args = parser.parse_args()

    # Unpack input_files and identify if csv or log
    input_files = args.input_files[:]

    # Parse in csv file of scan results
    if input_files[0].split('.')[-1] == 'csv':
        scan_results = pd.read_csv(args.input_files[0], index_col=0)
        if args.scan_param != None:
            scan_parameter = args.scan_param
        else:
            raise Exception("No scan parameter given, provide the header for the scan parameter column from csv as: -p [header]")

    # Create scan molecules and csv for scan log files
    else:
        scan_molecules, scan_info = ml.init_scan(*input_files, tracked_params=args.track_param_file)

        # Set scan parameter/s from scan info 
        if 'param_key' in scan_info.keys():
            scan_parameter = [scan_info['param_key']]
        else:
            scan_parameter = []
            for scan in scan_info.values():
                scan_parameter.append(scan['param_key'])
    
        # Set PES plot dimensionality from number of scan parameters
        args.plot = len(scan_parameter)

        # Create dataframe from results and save as csv
        scan_results = ml.mols_to_dataframe(scan_molecules, save='ps'+scan_parameter[0])
        print(scan_results)

    # Plot results of 1D or 2D PES (no plot if args.plot=0)
    if args.plot == 1:
        scan_results = scan_results.sort_values(scan_parameter)
        fig, ax = ml.plot_param_E(scan_results, parameter_col=scan_parameter[0], save=args.save, scan=True)
        plt.show()
    elif args.plot == 2:
        fig, ax = ml.plot_PES(scan_results, parameter_cols=scan_parameter, save=args.save, opt_filter=(args.opt_filter))
        plt.show()


