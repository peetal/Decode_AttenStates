# To reproduce results in Figure 2
## 1. Run `face_voxel_selection_cv.py`
Run this script for Perceive vs. Retrieve and Scramble vs. Retrieve comparisons, with different participants being the left-out-subject. The script takes 6 inputs: For more details please see my tutorial on FCMA (https://github.com/peetal/FCMA_demo), or checkout the BrainIAK website (https://brainiak.org/)
- the directory of the data, in this case would be the directory pointing to the 24 residual activity files (data/FIR_residual); 
- the suffix of the residual activity files, in this case would be "res1956.nii.gz";
- the directory of the shared gray matter mask file; 
- the directory of the epoch file; 
- the left-out-subject ID;
- the file output directory

This script outputs a NIFTI file of the same dimension of the residual activity file, named "fc_noXX_result_score.nii.gz". The value of each voxel indicating its ultility of differentiating the respective task condition comparison when using all the data except the XX participant. 
## 2. Run `create_composite_score.py`
This script computes a composite score for each voxel, indicating its ultility in separating perception from retrieval states, based on the respective fold's training data. The figure below summarize the two steps so far. 
<img width="509" alt="Screen Shot 2022-05-07 at 4 28 57 PM" src="https://user-images.githubusercontent.com/63365201/167275242-15ab4b42-9a35-4a4d-859b-e93786772dae.png">
## 3. Run `make_top_voxel_mask.sh`
This bash script selects the top performing voxels in terms of their composite score and create binary masks of the selected sizes. This script takes 3 inputs: 
- the directory of the data, in this case it would be the output directory of the create_composite_score file;
- the size of the voxel mask;
- the output directory
## 4. run `fcma_classify.py`
Run this script for each task condition comparison, with different participants being the left-out-subject. The script takes 8 inputs: 
- the directory of the data, in this case would be the directory pointing to the 24 residual activity files (data/FIR_residual); 
- the suffix of the residual activity files, in this case would be "res1956.nii.gz";
- the top_k_voxel_mask, in this case would be the output files of the make_top_voxel_mask.sh; 
- the directory of the epoch file, in this case there are 3 task condition comparisons, thus 3 epoch files (data/Epoch_file)
- the left-out-subject ID;
- the file output directory;
- idx: unrelated for the current task (leave it being 0);
- a string indicating the current task condition comparison;

This script outputs a csv file with the columns being 1) left-out-subject 2) classification accuracy (i.e., correct_num/32) 3) decision function output for each epoch (used later for computing AUC) 4) task condition comparison. Stacking all output CSVs leads to clf_results/bg_fcma_clf_conf.csv.
## 5. run `mvpa_classify.py`
Run this script for each task condition comparison, with different participants being the left-out-subject. This script does very similar things as the fcma_classify.py script, except that it trains and tests classifiers based on stimulus-evoked activity patterns instead of background FC patterns. The script taskes 7 inputs: 
- the directory of the data, in this case would be the directory pointing to the 24 stimulus-evoked time series (data/before_FIR); 
- the suffix of the stimulus-evoked time series, in this case would be "CONCAT.nii.gz";
- the top_k_voxel_mask, in this case would be the output files of the make_top_voxel_mask.sh;
- the directory of the epoch file, in this case there are 3 task condition comparisons, thus 3 epoch files (data/Epoch_file);
- the left-out-subject ID;
- a string indicating the current task condition comparison;

The output of this script resembles those of the fcma_classify.py script. Stacking all output CSVs leads to clf_results/evoked_mvpa_clf_conf.csv.
## 6. run `compute_AUC.py`
This script computes the area under the receiver operating curve (AUC) for each model given mask size (e.g., k = 3000), condition (e.g., ret vs. per), and neural measures (i.e., background FC, stimulus-evoked MVPA, and hybrid). Thus script outputs two files: clf_results/fcma_regular_AUC_full.csv and clf_results/fcma_mvpa_hybrid_3000AUC.csv, which could be used to reproduce Figure 2a and 2b in the manuscript. 

<img width="500" alt="image" src="https://user-images.githubusercontent.com/63365201/167278024-051e940d-ba93-47e3-b194-edbd686f5f2a.png">
