import os, sys, glob
# obtain full path for current directory: '.../rt-cloud/projects/sample'
currPath = os.path.dirname(os.path.realpath(__file__))
# obtain full path for root directory: '.../rt-cloud'
rootPath = os.path.dirname(os.path.dirname(currPath))

sys.path.append(rootPath)
from utils import check_dimension, read_nifti, resample_image

original_roi = sorted(glob.glob(os.path.join(currPath, "roi_2.5mm", "*.nii.gz")), key=lambda i: int(i.split("/")[-1].split("_")[0][3:]))
ref_img = '/Users/peetal/Documents/Github/rt-cloud/projects/phantom_test/data/standard/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'
for roi in original_roi: 

    out_name = os.path.join("/Users/peetal/Documents/Github/rt-cloud/projects/phantom_test/data/roi",
                            (roi.split("/")[-1].split(".")[0] + "_res-02.nii.gz"))
    roi_resampled = resample_image(roi, ref_img)
    roi_resampled.to_filename(out_name)



