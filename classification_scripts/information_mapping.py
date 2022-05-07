"""This script create a binary mask for voxels that capture state-related differences based on permutation test"""
##
## Author: Y.Peeta Li; 
## Email: peetal@uoregon.edu
##
import glob, os
from nilearn import image, masking 
import numpy as np 

# function for each iteration, compute the composite score for permuted data
def extract_voxel_mean(ret_per_3d, ret_scramble_3d):
    
    # load ret_per image
    ret_per_nii = image.load_img(ret_per_3d)
    ret_per_dat = ret_per_nii.get_fdata()
    # load ret_perscramble imgae
    ret_scramble_nii = image.load_img(ret_scramble_3d)
    ret_scramble_dat = ret_scramble_nii.get_fdata()

    # check dimension 
    if ret_per_dat.shape != ret_scramble_dat.shape:
        print("Shape does not match")
    else: 
        # stack the two 3d image on the 4th dimension
        stack_img = np.stack([ret_per_dat, ret_scramble_dat], axis = 3)
        # for each voxel, select the smaller number
        stack_img_mean = np.mean(stack_img, axis = 3)

    return(stack_img_mean)

# run script
if __name__ == '__main__':

    # --------------------
    # Permutate data 
    # --------------------
    # ret-per null data
    ret_per_dir = '/dir/to/ret_per_comparison/permutation/voxel_selection_output'
    ret_per_permutate = sorted(glob.glob(os.path.join(ret_per_dir, '*_score.nii.gz')), key= lambda x: int(x.split('_')[-3][8:]))

    # ret-scramble null data
    ret_scramble_dir = '/dir/to/scramble_ret_comparison/permutation/voxel_selection_output'
    ret_scramble_permutate = sorted(glob.glob(os.path.join(ret_scramble_dir, '*_score.nii.gz')), key= lambda x: int(x.split('_')[-3][8:]))

    # compute composite score image for each iteration 
    composite_score_3d = []
    for ret_per_3d, ret_scramble_3d in zip(ret_per_permutate, ret_scramble_permutate):
        composite_score_3d.append(extract_voxel_mean(ret_per_3d, ret_scramble_3d))

    # reference image
    ref = image.load_img('/dir/to/gray_matter_mask')
    # stack up 100 iterations: 
    permutation_composite_4d = np.stack(composite_score_3d, axis = 3)
    permutation_composite_4d_nii = image.new_img_like(ref, permutation_composite_4d, ref.affine)

    # --------------------
    # Real data
    # --------------------
    # load all actual data (composite score)
    inner_cv_dat_dir = '/dir/to/create_composite_score_output'
    inner_cv_dat = glob.glob(os.path.join(inner_cv_dat_dir, "*_score.nii.gz"))
    # stack them along the 4th axis
    inner_cv_dat_4d = np.stack([image.load_img(dat).get_fdata() for dat in inner_cv_dat], axis = 3)

    # averaged acc for each voxel
    inner_cv_dat_mean = np.mean(inner_cv_dat_4d, axis = 3)
    # mean and sd for each voxel for null distribution
    permutate_dat_mean = np.mean(permutation_composite_4d, axis = 3)
    permutate_dat_sd = np.std(permutation_composite_4d, axis = 3)

    # z score the observed data based on the null distribution 
    inner_cv_dat_zscore = (inner_cv_dat_mean - permutate_dat_mean)/permutate_dat_sd
    inner_cv_dat_zscore = np.nan_to_num(inner_cv_dat_zscore)
    inner_cv_dat_zscore_nii = image.new_img_like(ref, inner_cv_dat_zscore, ref.affine)
    inner_cv_dat_zscore_nii.to_filename('output_dir/inner_cv_composite_zscore.nii.gz')
