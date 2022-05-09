# print a short introduction on the internet window
print("""-----------------------------------------------------------------------------

initialize.py (Last Updated: 08/24/2021)
Author: Peeta Li @ peetal@uoregon.edu

The purpose of this script is to initialize the rt-cloud session. Specifically,
it will perform preprocessing on T1w image and created reference bold image in 
standard space. It would also config all necessary directories. 

-----------------------------------------------------------------------------""")

import os, sys, shutil, argparse, glob
from preprocessing_func import Struc, RefBOLD

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
from rtCommon.dataInterface import uploadFilesToCloud
from rtCommon.imageHandling import getTransform

# obtain the full path for the configuration toml file
defaultConfig = os.path.join(currPath, 'conf/phantom_test_local.toml')


def _initialize_image(cfg):
    
    # Only do MPRAGE if this IS the FIRST session for this subject, 
    if cfg.sessionNum == 1: 
        # pattern of structrual dicom files
        dicomScanNamePattern_struc = utils.stringPartialFormat(cfg.dicomNamePattern, 'SCAN', cfg.strucScanNum)
        struc_dicom_pattern = dicomScanNamePattern_struc.split('_')[0] + '_' + dicomScanNamePattern_struc.split('_')[1] + '_*' 
        print(struc_dicom_pattern)

        # preprocess structual image, including dicom2nifti, brain extraction, and segmentation. 
        sub_struc = Struc(struc_dicom_pattern, cfg)
        sub_struc.run_sub_struc_preprocessing()

    else: 
        print('Not the first session, skip structure preprocessing. Go directly to ref BOLD.')

    # create reference bold image and register it to standard space 
    # importantly, to get the func2standard and standard2func mat (linear) or warp (nonlinear)
    dicomScanNamePattern_refbold = utils.stringPartialFormat(cfg.dicomNamePattern, 'SCAN', cfg.refBoldScanNum)
    print(dicomScanNamePattern_refbold)
    sub_refbold = RefBOLD(dicomScanNamePattern_refbold, cfg)
    sub_refbold.func2standard_register()

    return None

def initialize_cfg(cfg):

    """
    |-- sub_dir
        |-- struc_dir 
            |-- sturc_timing.txt
            |-- fsl.bet_output 
            |-- fsl.fast_output
        |-- session_01
            |-- func_dir
                |-- ref_bold
                    |-- refbold.nii.gz
                    |-- standard2func_warp.nii.gz
                    |-- func2standard_warp.nii.gz
                    |-- temp
                        |-- halfway_files.nii.gz
                |-- roi_standard2func
                    |-- roi1_standard2func.nii.gz
                    |-- roi2_standard2func.nii.gz
                |-- run1                            # run folders will be created in the main script, not in initialization script. 
                    |-- raw_bold
                    |-- mc_bold
                    |-- mni_bold
                |-- run2                            # run folders will be created in the main script, not in initialization script.
                    |-- raw_bold
                    |-- mc_bold
                    |-- mni_bold
            |-- beh
                |-- refbold_timing.txt
                |-- run1                            # run folders will be created in the main script, not in initialization script. 
                    |-- roi_univariate.npy
                    |-- roi0_multivariate.npy
                    |-- vol_data
                        |-- tr_data.txt
                |-- run2                            # run folders will be created in the main script, not in initialization script. 
                    |-- roi_univariate.npy
                    |-- roi0_multivariate.npy
                    |-- vol_data
                        |-- tr_data.txt
    """

    def _check_dir(dir_list):

        for this_dir in dir_list: 

            if not os.path.isdir(this_dir): 
                os.makedirs(this_dir)
            else: 
                print(f"{this_dir} already exisit!")
        return None

    # Add new directorys
    cfg.sub_dir = os.path.join(cfg.codeDir, 'data', cfg.subjectName) # directory to store imaging data for a subject
    cfg.struc_dir = os.path.join(cfg.sub_dir, 'struc_nii')           # Only need to scan structural image for one time. 
    cfg.session_dir = os.path.join(cfg.sub_dir, f"session{cfg.sessionNum}")
    cfg.func_dir = os.path.join(cfg.session_dir, 'func_nii')         # function directory under each session 
    cfg.ref_bold = os.path.join(cfg.func_dir, 'ref_bold')            # reference bold for each session 
    cfg.ref_bold_temp_dir = os.path.join(cfg.ref_bold, 'temp')       # temp folder for storing temp files 
    cfg.roi_dir = os.path.join(cfg.func_dir, 'roi_standard2func')    # roi transformed onto functional space for this session. 
    cfg.beh = os.path.join(cfg.session_dir, 'beh')                   # where real time behavioral txt output files be stored. 
    _check_dir([cfg.sub_dir, cfg.struc_dir, cfg.session_dir, cfg.func_dir, cfg.ref_bold, cfg.ref_bold_temp_dir, cfg.beh, cfg.roi_dir])
    
    # check dicom directory, which depends on whether I'm at LCNI or testing local. 
    if cfg.location == "LCNI":
        dicomDir_pattern = str(cfg.imgDir) + '/' + str(cfg.datestr) + '.' + str(cfg.subjectName) + '*'
        cfg.dicomDir = glob.glob(dicomDir_pattern)[0]
        print("Location is at LCNI, the DICOM directory pattern is: " + cfg.dicomDir)
    elif cfg.location == "local":
        print("Location is at local, doing local testing")

    # add path to important transformation matrix for real time registration
    cfg.bold2standard_mat = os.path.join(cfg.ref_bold, 'func2standard.mat')
    cfg.boldref_img_native = os.path.join(cfg.ref_bold, 'refBOLD_avg.nii.gz')
    cfg.MNIstandard_brain = os.path.join(cfg.codeDir, 'data', 'standard', 'tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz')
    cfg.func2standard_warp = os.path.join(cfg.ref_bold, 'func2standard_warp.nii.gz')
    cfg.roi_standard2func_nonlinear = [os.path.join(cfg.roi_dir, roi_path.split('/')[-1]).split('.')[0] + '_2func_nonlinear.nii.gz' for roi_path in cfg.roi]
    cfg.roi_standard2func_linear = [os.path.join(cfg.roi_dir, roi_path.split('/')[-1]).split('.')[0] + '_2func_linear.nii.gz' for roi_path in cfg.roi]

    return(cfg)

    

def main(argv=None):
    """
    This is the main function that is called when you run 'intialize.py'.
    
    Here, you will load the configuration settings specified in the toml configuration 
    file, initiate the class dataInterface, and set up some directories and other 
    important things through 'initialize()'
    """

    # define the parameters that will be recognized later on to set up fileIterface
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                           help='experiment config file (.json or .toml)')
    args = argParser.parse_args(argv)

    # load the experiment configuration file
    cfg = utils.loadConfigFile(args.config)

    # establish the RPC connection to the projectInterface
    #clientInterface = ClientInterface()

    # now that we have the necessary variables, call the function 'initialize' in
    #   order to actually start reading dicoms and doing your analyses of interest!
    #   INPUT:
    #       [1] cfg (configuration file with important variables)
    #       [2] dataInterface (this will allow a script from the cloud to access files 
    #               from the stimulus computer)
    cfg = initialize_cfg(cfg)
    
    _initialize_image(cfg)
    #cfg = initialize_cfg(cfg)
    print("Initilization Succeed!")
    return 0


if __name__ == "__main__":
    """
    If 'initalize.py' is invoked as a program, then actually go through all of the 
    portions of this script. This statement is not satisfied if functions are called 
    from another script using "from initalize.py import FUNCTION"
    """
    main()
    sys.exit(0)
