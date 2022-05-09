from nilearn import image, masking
import os

def read_nifti(niftiFilename, get_data=True):

    if get_data: 
        output_file = image.load_img(niftiFilename).get_fdata()
    else: 
        output_file = image.load_img(niftiFilename)
    return output_file

def applymask(niftiFilename, mask):

    masked_array = masking.apply_mask(niftiFilename, mask)
    return masked_array

def check_dimension(image_path1, image_path2):

    sameShape = False
    image1_shape = image.load_img(image_path1).get_fdata().shape
    print(image1_shape)
    image2_shape = image.load_img(image_path2).get_fdata().shape
    print(image2_shape)

    if image1_shape == image2_shape: 
        sameShape = True
    return sameShape 

def resample_image(image_path1, image_path2):

    roi_image = image.load_img(image_path1)
    ref_image = image.load_img(image_path2)

    roi_resampled = image.resample_to_img(roi_image, ref_image, interpolation='nearest')
    
    return roi_resampled 

def getOutputFilename(runId, TRindex):
	""""Return station classification filename"""
	filename = "AverageAct_run-{0:02d}_TR-{1:03d}.txt".format(runId, TRindex)
	return filename


def config_run_dir(cfg, run_num):

    """
    Goal: if there are multiple runs within each session, the script will create a separate folder for each run, and sub-folders
          under that. 
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
    params: 
        cfg: cfg output from initilization 
        run_num: an integer, probably from enumerator. 
    """

    def _check_dir(dir_list):

        for this_dir in dir_list: 

            if not os.path.isdir(this_dir): 
                os.makedirs(this_dir)
            else: 
                print(f"{this_dir} already exisit!")
        return None
    
    # update for each run
    cfg.func_run_dir = os.path.join(cfg.func_dir, f"run{run_num}")
    cfg.raw_bold_dir = os.path.join(cfg.func_run_dir, 'raw_bold') # where raw real time nifti be stored
    cfg.mc_bold_dir = os.path.join(cfg.func_run_dir, 'mc_bold') # where motion corrected real time nifti be stored
    cfg.mni_bold_dir = os.path.join(cfg.func_run_dir, 'mni_bold') # where registered real time nifti be stored, if decided to do this step
    cfg.beh_run = os.path.join(cfg.beh, f"run{run_num}")
    cfg.beh_voldata = os.path.join(cfg.beh_run, "vol_data")

    _check_dir([cfg.func_run_dir, cfg.raw_bold_dir, cfg.mc_bold_dir, cfg.mni_bold_dir, cfg.beh_run, cfg.beh_voldata])

    return cfg