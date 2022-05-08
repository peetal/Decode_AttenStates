# Processed data
Due to the size of the processed data, we have uploaded both stimulus-evoked and residual timeseries to Open Science Framework: https://osf.io/yfwc7/
### Before_FIR 
This folder included the stimulus-evoked timeseries for each pariticiapnt. Note that all functional run were concatenated and z-scored, spatially smoothed and high-pass filter, and regressed out all confounds variables (sixe motion parameters and the mean signals from white matter and CSF).
### FIR_res
This filder included the residual timeseires for each participant. These timeseires were obtained by applying a finite impulse response (FIR) model to regress out the stimulus-evoked component from the stimulus-evoked timeseires. 
### all_sub_universal_GM_mask.nii.gz
This is the shared gray matter mask across all participants. 
### Epoch file 
Note that the order of task conditions were randomized across participants, meaning that the concatenated timeseries were not following a consistent order. Thus we included 3 epoch npy to tell the FCMA scripts which volumes of a participant is which condition. This folder also included two zip files for shuffled epoch labels for Retrieve vs. Perceive and Retreive vs. Scramble for the purpose of the permutation test and information mapping. 
### cluster_evoked_activity
Information mapping identified 16 clusters of different sizes. To facilitate subsequent analyses, I have extracted voxel-wise timeseries from each clsuter, for each participant, stored in a npy. This folder include all timeseries extracted from the stimulus-evoked activity. These data were used for conducting pattern similarity analyses in Figure 5.
### cluster_residual_activity
Same idea as above. This folder include all timeseries extracted from the residual activity. These data were used for network analyses and FC pattern analyses in Figure 4 and 5, respectively. 
