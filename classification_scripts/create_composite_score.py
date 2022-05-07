"""This script computes the voxel-wise composite score"""
"""This composite score indicates its ultility in capturing state-related differences"""
##
## Author: Y.Peeta Li; 
## Email: peetal@uoregon.edu
##
import os, glob
from nilearn import image, masking
import numpy as np 
from scipy.stats import rankdata

if __name__ == '__main__':
    # set up directory 
    ret_per_dir = '/dir/to/ret_per_comparison/voxel_selection_output'
    ret_per_sub = sorted(glob.glob(os.path.join(ret_per_dir,'*_score.nii.gz')), key= lambda x: int(x.split('_')[-3][2:]))
    ret_scramble_dir = '/dir/to/scramble_ret_comparison/voxel_selection_output'
    ret_scramble_sub = sorted(glob.glob(os.path.join(ret_scramble_dir,'*_score.nii.gz')), key= lambda x: int(x.split('_')[-3][2:]))
    out_dir = 'output/dir'

    # grey matter mask 
    gm_mask = image.load_img('/dir/to/gray_matter_mask')

    # loop through each subject, combine two 3d images 
    for sub in range(24):
        
        # load ret_per image
        ret_per_nii = image.load_img(ret_per_sub[sub])
        ret_per_dat = ret_per_nii.get_fdata()
        # load ret_perscramble imgae
        ret_scramble_nii = image.load_img(ret_scramble_sub[sub])
        ret_scramble_dat = ret_scramble_nii.get_fdata()
        
        # check dimension 
        if ret_per_dat.shape != ret_scramble_dat.shape:
            print("Shape does not match")
        else: 
            # stack the two 3d image on the 4th dimension
            stack_img = np.stack([ret_per_dat, ret_scramble_dat], axis = 3)
            # for each voxel, select the smaller number
            stack_img_min = np.min(stack_img, axis = 3)
            
            # write out score file
            stack_img_min_nii = image.new_img_like(gm_mask, stack_img_min, affine = gm_mask.affine)
            stack_img_min_nii.to_filename(os.path.join(out_dir, f'fc_no{sub}_ret_per_scramble_composite_score.nii.gz'))
            
            # write out seq file
            gm_voxels = masking.apply_mask(stack_img_min_nii, gm_mask)
            gm_voxels_order = rankdata(gm_voxels, method = 'ordinal')
            gm_voxels_seq = len(gm_voxels) - gm_voxels_order + 1
            gm_voxels_seq_nii = masking.unmask(gm_voxels_seq, gm_mask)
            
            gm_voxels_seq_nii.to_filename(os.path.join(out_dir, f'fc_no{sub}_ret_per_scramble_composite_seq.nii.gz'))
            
            