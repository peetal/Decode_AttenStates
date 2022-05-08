# To reproduce results in Figure 3
## 1. Run `permutate_fcma_voxel_selection.py`
The goal is to define chance-level ultiliy of each voxel in revealing state-level differences.
Here I shuffled the epoch labels of Perceive vs. Retrieve comparison (`../data/epoch_files/permutate_labels_ret_per`) and Scramble vs. Retrieve comparison (`../data/epoch_files/permutate_labels_scr_ret`). This script taskes 7 inputs: 
- the directory of the data, in this case would be the directory pointing to the 24 residual activity files (data/FIR_residual);
- the suffix of the residual activity files, in this case would be "res1956.nii.gz";
- the directory of the shared gray matter mask file;
- the directory of the **shuffled** epoch file (there should be 100 of them for each comparison); 
- the number indicating which shuffled epoch file to use 
- iteration: how many iterations to do per job. 
- the file output directory

Note that I did not do the outer-loop LOOCV here due to limits in computational power. Thus, all 24 participants were included in the inner loop and to examine the "ultility" of each voxel using the inner-loop LOOCV framework. 
## 2. Run `information_mapping.py`
This script does a couple things: 
- Compute a composite null score across the two comparisons for each voxel (by computing the mean) 
- Compute the mean and standard diviation (across the 100 iteration) of each voxel 
- Compute the averaged observed data (across 24 real data) 
- Z-score the observed data using the mean and SD of the null distribution. 
- This script outputs `/z-scored_observed_data/inner_cv_composite_zscore.nii.gz`, each voxel contains the z-score indicating how whether this voxel's utility in separating state-related difference is significantly above chance. 
## 3. Using AFNI 3dClusterize to clsuter spatially contiguous voxels
Run AFNI command: `3dClusterize -inset inner_cv_composite_zscore.nii.gz -mask data/all_sub_universal_GM_mask.nii.gz -ithr 0 -1sided RIGHT_TAIL 3.7 -NN 2 -pref_map afni_cluster -summarize`. 
This command would first select voxels that pass a threshold (z > 3.7, which corresponds to p < 0.0001, right-tail). Then it would cluster spatially contiguous voxels. This command resulted in 62 clusters as shown in `afni_clus_info.txt`. Cluster selection was conducted to select the top 16 clusters (in terms of voxel size), as described in text (just more model trianing/testing). Below shows the 16 clustes, the mask for each cluster is also provided here. 

<img width="714" alt="Screen Shot 2022-05-07 at 11 26 24 PM" src="https://user-images.githubusercontent.com/63365201/167284683-aab009c3-9668-41d7-95d5-d800e6f7a335.png">

## 4. Go through `force-directed-plots.ipynb`
This notebook does a couple things: 
- Parse residual acitivties of each cluster, each particiapnt into background FC matrices for each peoch, each participant. 
- Constructs a graph for each condition. 
- Run Louvain community detection algorithm 1000 times to find the partition that maximizes graph modularities. 
- Project the functional relevance between each functional communitiy to a force-directed plot. Note that these plots are stochastic, meaning that they won't be identical everytime you run it. But the general structure should be identical most of the times.  

<img width="712" alt="Screen Shot 2022-05-07 at 11 31 43 PM" src="https://user-images.githubusercontent.com/63365201/167284805-36bcf614-f8d1-484a-a492-b32491dae565.png">
