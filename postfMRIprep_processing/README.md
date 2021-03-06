# Finite impulse response model 
The FIR model was used to model stimulus-evoked component of the BOLD timeseries. The fMRIPrep minimally preprocessed timeseries went through the following steps to extrat the stimulus-free, residual activities. 
## `pl_post_fmriprep.py`
Spatial smoothing and high-pass filtering: 5.0 mm FWHM Gaussian kernel and high-pass filtered at 0.01 Hz.
## `FIR_glm.py` (implemented using Nipype; see the flowchart below)
1. Creating a shared (i.e., group-level) gray matter voxel masks 
2. GLM1: Nuisance regressors. Removing six motion parameters and the mean signals from CSF and WM. 
3. Concatenate and Zscore: Concatenate all functional runs and zscore each volume using the mean and SD of the IBI timepoints. 
4. GLM2: Finite impulse response model. Modeled the first 36 TRs for every epoch separately for face and scene epochs, resulting in 36 (TR) x 2 (epoch category) x 3 (condition) = 216 regressors. FIR is believed to be the optimal GLM for removing task-evoked response because it does not assume the shape of the hemodynamic response function. A FIR design matrix may look like the following, there are in a total of 1956 rows, standing for 1956 TR/volume and 216 columns, standing for 216 regressors. Each regressor would be either 0 or 1 for each TR. 
<img src="https://user-images.githubusercontent.com/63365201/167275836-39bf3f4e-e789-48a9-9a75-8fca1168b10e.png" width="1000" height="800">
## Outputs of FIR_glm.py: 
1. Residual timeseries after the first GLM (i.e., after regressing out confounds: data/before_FIR). 
2. Residual timeseries after the second GLM (i.e.., after regressing stimulus-evoked components: data/FIR_res). 
<img src="https://user-images.githubusercontent.com/63365201/166166948-3820b0cc-1eb7-4a17-a31f-e16386a2c794.PNG" width="500" height="650">
