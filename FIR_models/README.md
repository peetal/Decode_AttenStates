# Finite impulse response model 
The FIR model was used to model stimulus-evoked component of the BOLD timeseries. The fMRIPrep minimally preprocessed timeseries went through the following steps to extrat the stimulus-free, residual activities. 
## Spatial smoothing and high-pass filtering 
5.0 mm FWHM Gaussian kernel and high-pass filtered at 0.01 Hz. See pl_post_fmriprep.py for details
![IMG_0052](https://user-images.githubusercontent.com/63365201/166166414-4d157238-0081-417b-9f2f-dfdefa45584e.png)
