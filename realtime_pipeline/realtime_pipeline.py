"""-----------------------------------------------------------------------------

realtime_pipeline.py (Last Updated: 08/24/2021)
Author: Peeta Li
-----------------------------------------------------------------------------"""

verbose = False
useInitWatch = True

# import important modules
import os
import sys
import argparse
import warnings
import time
import numpy as np
import nibabel as nib
import scipy.io as sio
#import keyboard as kb
import json
from initialize import initialize_cfg
from preprocessing_func import ProcessNewVol
from utils import read_nifti, getOutputFilename, applymask, config_run_dir

with warnings.catch_warnings():
    if not verbose:
        warnings.filterwarnings("ignore", category=UserWarning)
    from nibabel.nicom import dicomreaders

# obtain full path for current directory: '.../rt-cloud/projects/sample'
currPath = os.path.dirname(os.path.realpath(__file__))
# obtain full path for root directory: '.../rt-cloud'
rootPath = os.path.dirname(os.path.dirname(currPath))

# add the path for the root directory to your python path so that you can import
#   project modules from rt-cloud
sys.path.append(rootPath)
# import project modules from rt-cloud
from rtCommon.utils import loadConfigFile, stringPartialFormat
from rtCommon.clientInterface import ClientInterface
from rtCommon.imageHandling import readRetryDicomFromDataInterface, convertDicomImgToNifti




def doRuns(cfg, dataInterface, subjInterface, webInterface):
    """
    This function is called by 'main()' below. Here, we use the 'dataInterface'
    to read in dicoms (presumably from the scanner, but here it's from a folder
    with previously collected dicom files), doing some sort of analysis in the
    cloud, and then sending the info to the web browser.

    INPUT:
        [1] cfg - configuration file with important variables)
        [2] dataInterface - this will allow this script runnin in the cloud to access
                files from the stimulus computer, which receives dicom files directly
                from the MRI Scanner console
        [3] subjInterface - this allows sending feedback (e.g. classification results)
                to a subjectService running on the presentation computer to provide
                feedback to the subject (and optionally get their response).
        [4] webInterface - this allows updating information on the experimenter webpage.
                For example to plot data points, or update status messages.
    OUTPUT:
        None.
    """
    # cfg.scanNum is a list of scan number. 
    for runNum, scanNum in zip(cfg.runNum, cfg.scanNum):

        print(f"Doing run {runNum}, which is scan {scanNum}")

        # set up directory for this run, creating run dir, and subdirs under that run 
        cfg = config_run_dir(cfg, runNum)

        # the dicom pattern for incoming dicoms 
        dicomScanNamePattern = stringPartialFormat(cfg.dicomNamePattern, 'SCAN', scanNum)
        # set up for watchdog 
        streamId = dataInterface.initScannerStream(cfg.dicomDir,
                                                dicomScanNamePattern,
                                                cfg.minExpectedDicomSize)

        # preparing for plotting in webinterface 
        webInterface.clearRunPlot(runNum)

        # total number of TR for this run
        num_total_TRs = cfg.numTR_func_run  # number of TRs to use 
        
        # for storing the ROI multivariate activation patterns and univariate activation measure 
        roi_num = len(cfg.roi_standard2func_nonlinear)
        roi_volumes = [np.sum(read_nifti(roi_file)) for roi_file in cfg.roi_standard2func_nonlinear] # the size of each roi 
        roi_niimg = [read_nifti(roi_file, get_data = False) for roi_file in cfg.roi_standard2func_nonlinear] # read roi as niimg
        
        roi_avg_activations = np.zeros((roi_num, num_total_TRs)) # stored the ROI averaged activation. 
        activation_pattern = [np.zeros((num_total_TRs, int(roi_vol))) for roi_vol in roi_volumes] # initialize a list of the size num_roi, each item is an array of the size (tr, roi_vol)

        timing_info = []
        for this_TR in list(range(1, num_total_TRs+1)):
        
            # declare variables that are needed to use in get data requests
            timeout_file = 180 # small number because of demo, can increase for real-time
            dicomFilename = dicomScanNamePattern.format(TR=this_TR)

            # get dicome file 
            print(f'Processing TR {this_TR}')
            dicomData = dataInterface.getImageData(streamId, int(this_TR), timeout_file)

            if dicomData is None:
                print('Error: getImageData returned None') 
                return
            
            # dcm2nii and converting new nifti to mni template 
            curTR_nifti = ProcessNewVol(dicomFilename, cfg)
            #nifti_reg = curTR_nifti.register_new_nifti(mc = True) # Do not want to register on to MNI 
            nifti_mc = curTR_nifti.dcm2nii_and_mc() # Just do things on the functional space.  
            
            # read in the new nifti file 
            niftiObject = read_nifti(nifti_mc, get_data=False)

            # take the average of ROI activation values
            roi_patterns = [applymask(niftiObject, niimg) for niimg in roi_niimg]
            roi_act_value = [np.mean(roi_pattern) for roi_pattern in roi_patterns] # one value fOR each ROI
            roi_avg_activations[:,this_TR-1] = roi_act_value

            # the activation pattern for each roi in the list 
            for pattern, roi in zip(activation_pattern, roi_patterns):
                pattern[this_TR-1,:] = roi[:,] 
            #activation_pattern = [pattern[this_TR-1, roi[:,]] for pattern, roi in zip(activation_pattern, roi_patterns)]

            # avg_niftiData = np.round(avg_niftiData,decimals=2)
            print(f"The average ROI activation value for TR {this_TR} is {roi_act_value}")
            
            # create real time txt output file, and save for each tr 
            text_to_save = '{0:.2f}'.format(roi_act_value[0]) 
            file_name_to_save = getOutputFilename(scanNum, this_TR) # naming
            full_file_name_to_save = os.path.join(cfg.beh_voldata, file_name_to_save)
            try:
                dataInterface.putFile(full_file_name_to_save, text_to_save)
            except Exception as err:
                print('Error putFile: ' + str(err))
                return

            # plot on web interface 
            webInterface.plotDataPoint(runNum, int(this_TR), float(roi_act_value[0]))

            tr_done_timing = time.time() # timing for finish processing the current tr
            timing_info.append({f'TR_{this_TR} done time':  tr_done_timing})

        # save out timing info for this run
        timing_info_outputdir = os.path.join(cfg.beh_run, 'timinginfo.txt')

        with open(timing_info_outputdir, 'w') as f:
            json.dump(timing_info, f)

        # save univariate and multivariate data for each ROI 
        np.save(os.path.join(cfg.beh_run, 'roi_univaraite.npy'), roi_avg_activations) # (roiNum, trNum)
        for idx, data in enumerate(activation_pattern):
            np.save(os.path.join(cfg.beh_run, f'roi{idx}_multivariate.npy'), data) # the idx is based on toml input, of the shape (trNum, voxNum)

    return


def main(argv=None):
    global verbose, useInitWatch
    """
    This is the main function that is called when you run 'sample.py'.

    Here, you will load the configuration settings specified in the toml configuration
    file, initiate the clientInterface for communication with the projectServer (via
    its sub-interfaces: dataInterface, subjInterface, and webInterface). Ant then call
    the function 'doRuns' to actually start doing the experiment.
    """

    # Some generally recommended arguments to parse for all experiment scripts
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c', type=str,
                           help='experiment config file (.json or .toml)')
    argParser.add_argument('--runs', '-r', default=None, type=str,
                           help='Comma separated list of run numbers')
    argParser.add_argument('--scans', '-s', default=None, type=str,
                           help='Comma separated list of scan number')
    argParser.add_argument('--yesToPrompts', '-y', default=False, action='store_true',
                           help='automatically answer tyes to any prompts')

    # Some additional parameters only used for this sample project
    argParser.add_argument('--useInitWatch', '-w', default=False, action='store_true',
                           help='use initWatch() functions instead of stream functions')
    argParser.add_argument('--noVerbose', '-nv', default=False, action='store_true',
                           help='print verbose output')

    args = argParser.parse_args(argv)

    useInitWatch = args.useInitWatch
    verbose = not args.noVerbose

    # load the experiment configuration file
    cfg = loadConfigFile(args.config)

    # Initialize the RPC connection to the projectInterface.
    # This will give us a dataInterface for retrieving files,
    # a subjectInterface for giving feedback, and a webInterface
    # for updating what is displayed on the experimenter's webpage.
    clientInterfaces = ClientInterface(yesToPrompts=args.yesToPrompts, rpyc_timeout=180)
    dataInterface = clientInterfaces.dataInterface
    bidsInterface = clientInterfaces.bidsInterface
    subjInterface = clientInterfaces.subjInterface
    webInterface  = clientInterfaces.webInterface

    # obtain paths for important directories (e.g. location of dicom files)
    #if cfg.imgDir is None:
    #    cfg.imgDir = os.path.join(currPath, 'dicomDir')
    cfg = initialize_cfg(cfg)

    # now that we have the necessary variables, call the function 'doRuns' in order
    #   to actually start reading dicoms and doing your analyses of interest!
    #   INPUT:
    #       [1] cfg (configuration file with important variables)
    #       [2] dataInterface (this will allow a script from the cloud to access files
    #            from the stimulus computer that receives dicoms from the Siemens
    #            console computer)
    #       [3] subjInterface - this allows sending feedback (e.g. classification results)
    #            to a subjectService running on the presentation computer to provide
    #            feedback to the subject (and optionally get their response).
    #       [4] webInterface - this allows updating information on the experimenter webpage.
    #            For example to plot data points, or update status messages.
    doRuns(cfg, dataInterface, subjInterface, webInterface)
    return 0


if __name__ == "__main__":
    """
    If 'sample.py' is invoked as a program, then actually go through all of the portions
    of this script. This statement is not satisfied if functions are called from another
    script using "from sample.py import FUNCTION"
    """
    main()
    sys.exit(0)
