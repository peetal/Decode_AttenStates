import os
import sys
import struct
import logging
import argparse
import numpy as np
import scipy.io as sio
from subprocess import call
from utils import read_nifti
from initialize import initialize_cfg

# obtain full path for current directory: '.../rt-cloud/projects/sample'
currPath = os.path.dirname(os.path.realpath(__file__))
# obtain full path for root directory: '.../rt-cloud'
rootPath = os.path.dirname(os.path.dirname(currPath))

# add the path for the root directory to your python path so that you can import
#   project modules from rt-cloud
sys.path.append(rootPath)
# import project modules from rt-cloud
import rtCommon.utils as utils
from rtCommon.clientInterface import ClientInterface
from rtCommon.dataInterface import downloadFolderFromCloud

# obtain the full path for the configuration toml file

def finalize(cfg):

    all_runs = cfg.runNum

    # locate the mc_bold folder for each run 
    mc_bold_dir = [os.path.join(cfg.func_dir, f"run{run}", 'mc_bold') for run in all_runs]
    out_file = [os.path.join(cfg.func_dir, f"run{run}", f"rt_preprocessed4D_run{run}.nii.gz") for run in all_runs]

    for in_dir, out_dir in zip(mc_bold_dir, out_file):
        command = 'fslmerge -t {0} {1}'.format(out_dir, os.path.join(in_dir, "*.nii.gz"))
        call(command,shell=True)

        # check dim
        check_img = read_nifti(out_dir, get_data=True)
        if check_img.shape[3] == cfg.numTR_func_run:
            print(f"{out_dir} shape is good!")
        else: 
            print(f"{out_dir} shape is wrong!")
    return None

    
def main(argv=None):
    """
    This is the main function that is called when you run 'finalize.py'.
    
    Here, you will load the configuration settings specified in the toml configuration 
    file, initiate the class dataInterface, and set up some directories and other 
    important things through 'finalize()'
    """

    # define the parameters that will be recognized later on to set up fileIterface
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c',type=str,
                           help='experiment config file (.json or .toml)')
    args = argParser.parse_args(argv)

    # load the experiment configuration file
    cfg = utils.loadConfigFile(args.config)
    cfg = initialize_cfg(cfg)


    # now that we have the necessary variables, call the function 'finalize' in
    #   order to actually start reading dicoms and doing your analyses of interest!
    #   INPUT:
    #       [1] cfg (configuration file with important variables)
    #       [2] dataInterface (this will allow a script from the cloud to access files 
    #               from the stimulus computer)
    finalize(cfg)
    return 0


if __name__ == "__main__":
    """
    If 'finalize.py' is invoked as a program, then actually go through all of the 
    portions of this script. This statement is not satisfied if functions are called 
    from another script using "from finalize.py import FUNCTION"
    """
    main()
    sys.exit(0)
