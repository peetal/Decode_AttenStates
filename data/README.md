# To reproduce results in Figure 2
## 1. Run face_voxel_selection_cv.py 
Run this script for Perceive vs. Retrieve and Scramble vs. Retrieve comparisons, with different participants being the left-out-subject. The script takes 6 inputs: For more details please see my tutorial on FCMA (https://github.com/peetal/FCMA_demo), or checkout the BrainIAK website (https://brainiak.org/)
- the directory of the data, in this case would be the directory pointing to the 24 residual activity files (data/sub-xx/FIR_residual); 
- the suffix of the residual activity files, in this case would be "res1956.nii.gz";
- the directory of the shared gray matter mask file; 
- the directory of the epoch file; 
- the left-out-subject ID;
- the file output directory
This script outputs a NIFTI file of the same dimension of the residual activity file, named "fc_noXX_result_score.nii.gz". The value of each voxel indicating its ultility of differentiating the respective task condition comparison when using all the data except the XX participant. 
## 2. Run
