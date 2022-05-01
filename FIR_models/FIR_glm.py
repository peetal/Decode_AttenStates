import pandas as pd
import numpy as np
import os
import glob
from nipype.interfaces.base import Bunch
from nilearn import image
from nipype.pipeline import engine as pe
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.algorithms.modelgen import SpecifyModel
from nilearn import image
import argparse

parser = argparse.ArgumentParser(description='First regress out confounds using FEAT, then do FIR modeling')
parser.add_argument(
    '--base_dir', 
    action='store', 
    default='/projects/hulacon/shared/strongbad01/strongbad01/subjects',
    help='The directory that stores smoothing and highpass resulted files.')

parser.add_argument(
    '--sub-id',
    required=True,
    action='store',
    help='subject id, prefix do not include sub- ')

parser.add_argument(
    '--n_procs',
    action='store',
    default=16,
    type=int,
    help='Maximum number of threads across all processes.')
args = parser.parse_args()

# -----------------
# Basic information
# -----------------
# per subject information
subj_id = args.sub_id
task_id = 'SB'
run_id = ['{:02d}'.format(x) for x in range(1,7)]
# directory information
base_dir = args.base_dir
output_dir = os.path.join(base_dir,'FIR_glm_model')
log_dir = os.path.join(base_dir,'FIR_glm_model')
# nproc
n_proc = args.n_procs


# -----------------
# Helper functions
# -----------------

# scale up the whole image intensity value by making the median = 10000
# This is to fix the potential problem of FEAT when dealing with close to 0 value 
def _get_inormscale(median_values):
    if isinstance(median_values, float):
        op_string = '-mul {:.10f}'.format(10000 / median_values)
    elif isinstance(median_values, list):
        op_string = ['-mul {:.10f}'.format(10000 / val) for val in median_values]
    return op_string

# create the trash EV (3 columns format) and 8 nuisance regressors for a subject
# Intotal 9 EVs
def creat_subject_info(regressors_file):
    from nipype.interfaces.base import Bunch
    import pandas as pd
    import numpy as np
    confounds = pd.read_csv(regressors_file)
    conf_name = confounds.columns
    conf_value = []
    for con in conf_name:
        value = confounds[f'{con}'].tolist()
        conf_value.append(value)
    
    subject_info = Bunch(
            conditions=['trash_ev'],
            onsets=[[0]],
            durations=[[0]],
            amplitudes=[[0]],
            regressor_names=conf_name,
            regressors=conf_value,
            tmod=None,
            pmod=None)
    
    return subject_info

# Zscore each seperate image using the mean and sd of the resting period. 
def zscore_withRestingPeriod(func, mask, subj_id, run_id):
    
    from nilearn import image, masking
    import pandas as pd
    import numpy as np
    import os
    import argparse
    from scipy import stats
    from itertools import chain, product
    
    np.seterr(divide='ignore', invalid='ignore')
    
    # generate the volume index for resting period.
    block1 = range(0,32) # block1: 28 + 4, b/c the 12 resting volumes are shifted backward for 4 volumes
    resting1 = range(32,44)
    block2 = range(44,72)
    resting2 = range(72,84)
    block3 = range(84,112)
    resting3 = range(112,124)
    block4 = range(124,152)
    resting4 = range(152,164)
    block5 = range(164,192)
    resting5 = range(192,204)
    block6 = range(204,232)
    resting6 = range(232,244)
    block7 = range(244,272)
    resting7 = range(272,284)
    block8 = range(284,312)
    resting8 = range(312,326) # 12 + 2 aftering shifting backward for 4 volumes, the last 6 dummy scans had 2 left
    
    # get the volume index for resting period and task period
    task = list(chain(block1, block2,  block3, block4,  block5, block6, block7, block8))
    resting = list(chain(resting1, resting2, resting3, resting4, resting5, resting6, resting7, resting8))
    
    # load func nifti 326 volumes
    img = image.load_img(func)
    # load the corresponding mask, with 0 and 1. 
    mask = image.load_img(mask)
    
    # mask the brain
    img_brain = masking.apply_mask(img,mask)
    # get resting volumes 
    img_brain_resting = img_brain[resting,:]
    
    # sd of the resting volumes
    img_brain_resting_sd = np.std(img_brain_resting, axis = 0, ddof = 1)
    
    # check dimension
    if img_brain_resting_sd.shape[0] == np.sum(mask.get_data()[:,:,:]): 
        # check if there is any voxel that have SD = 0 across all resting volumes
        if (img_brain_resting_sd == 0).any():
            index_tup = np.where(img_brain_resting_sd == 0)
            for i in range(len(index_tup)):
                img_brain_resting_sd[index_tup[i][0]] = np.mean(img_brain_resting_sd)
        
        # compute the mean and sd image for the resting period in te 4D structure
        img_resting = img.get_data()[:,:,:,resting]
        img_resting_mean = np.mean(img_resting, axis = 3)
        img_resting_sd = np.std(img_resting, axis = 3)

        # convert the one volume mean image into a 4d image, with 326 volumes
        resting_mean_4d_unstack = [img_resting_mean for _ in range(326)]
        resting_mean_4d_stack = np.stack(resting_mean_4d_unstack, axis = 3)
        # convert the one volume sd image into a 4d image, with 326 volumes
        resting_sd_4d_unstack = [img_resting_sd for _ in range(326)]
        resting_sd_4d_stack = np.stack(resting_sd_4d_unstack, axis = 3)

        # compute the z_score image (using resting period mean and sd)
        img_resting_centered = (img.get_data() - resting_mean_4d_stack)/resting_sd_4d_stack 

        # times the z_scored image with the mask, so non-brain voxels become 0.
        # first make the mask into 4d structure
        mask_4d_unstack = [mask.get_data() for _ in range(326)]
        mask_4d_stack = np.stack(mask_4d_unstack, axis = 3)
        # then element-wise multiply mask and time-series arrays
        zscore_brain_array = np.multiply(img_resting_centered, mask_4d_stack)
        # change all NAs to 0
        zscore_brain_array[np.isnan(zscore_brain_array)] = 0

        # write the z score array into a nifti file with the same header 
        img2 = image.new_img_like(img,zscore_brain_array, copy_header = True) 

        img2.to_filename(os.path.join(os.getcwd(),f'sub-{subj_id}_{run_id}_nuisanceRES_Zscore.nii.gz'))

        res_file_demean = os.path.join(os.getcwd(),f'sub-{subj_id}_{run_id}_nuisanceRES_Zscore.nii.gz')

    else: 
        res_file_demean = False
        
    return(res_file_demean)

# merge the 6 runs of a person after zscoring each image seperately, rename the resulted file
def _output_merge_filename(in_file, subj_id):
    
    import os
    from shutil import copyfile 
    
    out_fn = os.path.join(os.getcwd(),
                         (f'sub-{subj_id}_nuisanceRES_CONCAT.nii.gz'))
    
    copyfile(in_file, out_fn)
    return out_fn


# create a shared mask for 6 runs. Only the shared brain voxels = 1, others = 0
def create_shared_mask(mask_list, subj_id):
    
    import os
    import numpy as np
    from nilearn import image
    
    # create a shared mask template with a mask dimension and all 1s
    shared_mask = np.ones([78,92,78])
    # mask_list is a list of 6 brain masks for 6 runs from fmriprep
    for mask in mask_list: # mask = masks_list[0]
        # read in a single brain mask
        mask_img = image.load_img(mask)
        mask_array = mask_img.get_data()
        # multiply the shared mask template with the current brain mask, make non-brain voxels 0
        shared_mask = np.multiply(shared_mask, mask_array)
    
    # write the array to the niming like object
    shared_mask_img = image.new_img_like(mask_img,shared_mask) 
    # write out the shared brainmask
    shared_mask_img.to_filename(os.path.join(os.getcwd(),f'sub-{subj_id}_run-shared_brain-mask.nii.gz'))
    # output the path pointing out to the shared brainmask
    shared_mask_img_path = os.path.join(os.getcwd(),f'sub-{subj_id}_run-shared_brain-mask.nii.gz')
    
    return shared_mask_img_path

# for datasink naming 
def _perfix_id(subj_id):
    
    return f'sub-{subj_id}'

# for FIR glm output beta naming
def _beta_naming(subj_id):
     
    import os
    return os.path.join(os.getcwd(), (f'sub-{subj_id}_beta216.nii.gz'))

# for FIR glm output residual naming
def _res_naming(subj_id):
    
    import os
    return os.path.join(os.getcwd(), (f'sub-{subj_id}_res1956.nii.gz'))

# ---------------------------------------------------
# Datasource - func and reg design_mat and brainmask
# ---------------------------------------------------
os.makedirs(os.path.join(output_dir,'working',subj_id))
wf = pe.Workflow(name = 'glm_model')
wf.base_dir = os.path.join(output_dir,'working',subj_id)

# Feed the dynamic parameters into the node
inputnode = pe.Node(
    interface = niu.IdentityInterface(
    fields = ['subj_id','task_id','run_id']), name = 'inputnode')
# specified above
inputnode.inputs.subj_id = subj_id
inputnode.inputs.task_id = task_id
inputnode.inputs.run_id = run_id

# grab input data
datasource = pe.MapNode(
    interface = nio.DataGrabber(
        infields = ['subj_id','task_id','run_id'],
        outfields = ['func','regressor','brainmask']),
        iterfield = ['run_id'],
        name = 'datasource')
# Location of the dataset folder
datasource.inputs.base_directory = base_dir  
# Necessary default parameters 
datasource.inputs.template = '*' 
datasource.inputs.sort_filelist = True
# the string template to match
# field_template and template_args are both dictionaries whose keys correspond to the outfields keyword
datasource.inputs.template_args = dict(
    func=[['subj_id', 'subj_id', 'task_id', 'run_id']],
    regressor =[['subj_id', 'task_id', 'run_id']],
    brainmask = [['subj_id', 'subj_id', 'task_id', 'run_id']])
datasource.inputs.field_template = dict(
    func = 'post_fmriprep/sub-%s/sub-%s_task-%s_run-%s_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz',
    regressor = 'selected_confounds/sub-%s_task-%s_run-%s_desc-confounds_regressors.csv',
    brainmask = 'derivative/fmriprep/sub-%s/func/sub-%s_task-%s_run-%s_space-MNI152NLin2009cAsym_res-native_desc-brain_mask.nii.gz')

# grab design file, this is a Node, not MapNode
designsource = pe.Node(
    interface = nio.DataGrabber(
        infields = ['subj_id'],
        outfields = ['design_mat']),
    name = 'designsource'
)
# Location of the dataset folder
designsource.inputs.base_directory = base_dir  
# Necessary default parameters 
designsource.inputs.template = '*' 
designsource.inputs.sort_filelist = True
designsource.inputs.template_args = dict(
    design_mat = [['subj_id']])
designsource.inputs.field_template = dict(
    design_mat = 'SB_FIR_design/sub-%s_FIR_design.txt')

# Datasink for future use
datasink = pe.Node(interface = nio.DataSink(), name = 'datasink')
datasink.inputs.base_directory = output_dir
datasink.inputs.parameterization = False


# connect the input node with datagrabber, so the datagrabber will use the dynamic parameters to grab func and brain mask for each run
wf.connect([(inputnode, datasource, [('subj_id', 'subj_id'),
                                        ('task_id', 'task_id'),
                                        ('run_id', 'run_id')]),
            (inputnode, designsource, [('subj_id', 'subj_id')]),
            (inputnode, datasink, [(('subj_id', _perfix_id), 'container')])
])

# -----------
# Pre-FEAT
# -----------

# Get median value for each EPI data
median_values = pe.MapNode(
    interface=fsl.ImageStats(op_string='-k %s -p 50'),
    iterfield = ['in_file','mask_file'],
    name='median_values')

# Normalize each EPI data intensity to 10000
intensity_norm = pe.MapNode(
    interface=fsl.ImageMaths(suffix='_intnorm'),
    iterfield = ['in_file','op_string'],
    name='intensity_norm')

# Mask functional image
masked_func = pe.MapNode(
    interface=fsl.ApplyMask(),
    iterfield = ['in_file','mask_file'],
    name='skullstrip_func')

wf.connect([(datasource, median_values,[('func', 'in_file'),
                                        ('brainmask', 'mask_file')]),
            (datasource, intensity_norm, [('func','in_file')]),
            (median_values, intensity_norm, [(('out_stat', _get_inormscale), 'op_string')]),
            (intensity_norm, masked_func, [('out_file', 'in_file')]),
            (datasource, masked_func, [('brainmask', 'mask_file')])
])

# ---------------
# Set up FEAT
# ---------------

# Generate nuisance regressor and 1 trash EV (last TR)
create_regressor = pe.MapNode(
interface=niu.Function(
    input_names=['regressors_file'],
    output_names=['subject_info'],
    function=creat_subject_info),
        iterfield =['regressors_file'], name='create_regressor')

# feed in the nuisance regressor and EV
model_spec = pe.MapNode(
    interface=SpecifyModel(),
    iterfield=['subject_info'],
    name='create_model_infos')
model_spec.inputs.input_units = 'secs'
model_spec.inputs.time_repetition = 1 # TR
model_spec.inputs.high_pass_filter_cutoff = 100 # high-pass the design matrix with the same cutoff 

# Generate fsf and ev files for the nuisance GLM
design_nuisance = pe.MapNode(
    interface=fsl.Level1Design(),
    iterfield=['session_info'],
    name='create_nuisance_design')
design_nuisance.inputs.interscan_interval = 1 # TR
design_nuisance.inputs.bases = {'dgamma': {'derivs': False}} # convolution, ideally should be none, but idk how. For tash ev, should not matter much
design_nuisance.inputs.model_serial_correlations = True # prewhitening

# Generate the contrast and mat file for the nuisance GLM
model_nuisance = pe.MapNode(
    interface=fsl.FEATModel(),
    iterfield=['fsf_file', 'ev_files'],
    name='create_nuisance_model')

# Estimate nuisance GLM model
model_estimate = pe.MapNode(
    interface=fsl.FILMGLS(),
    iterfield=['in_file','design_file', 'tcon_file'],
    name='estimate_nuisance_models')
model_estimate.inputs.smooth_autocorr = True
model_estimate.inputs.mask_size = 5
model_estimate.inputs.threshold = 1000

wf.connect([
    (datasource, create_regressor, [('regressor','regressors_file')]),
    (create_regressor, model_spec, [('subject_info','subject_info')]),
    (masked_func, model_spec, [('out_file','functional_runs')]),
    (model_spec, design_nuisance, [('session_info','session_info')]),
    (design_nuisance, model_nuisance, [('ev_files', 'ev_files'),
                                       ('fsf_files', 'fsf_file')]),
    (model_nuisance, model_estimate, [('design_file', 'design_file'),
                                      ('con_file', 'tcon_file')]),
    (masked_func, model_estimate, [('out_file','in_file')])
])

# ---------------
# Post-FEAT
# ---------------

# Z score each image based on resting period mean and SD
# THIS IS THE END OF MAPNODE!
z_score = pe.MapNode(
interface=niu.Function(
    input_names=['func','mask','subj_id','run_id'],
    output_names=['res_file_demean'],
    function=zscore_withRestingPeriod),
    iterfield = ['func','mask','run_id'],
name='zscore')

# Concatenate 6 z_scored files
concatenate = pe.Node(interface = fsl.Merge(), name = 'merge_res')
concatenate.inputs.dimension = 't'

# rename the Nuisance_regressed_Zscored_Concatenated file
rename_merge = pe.Node(
    interface = niu.Function(
        input_names = ['in_file','subj_id'],
        output_names = ['out_file'],
        function = _output_merge_filename),
    name = 'rename_merge')

wf.connect([
    (model_estimate,z_score,[('residual4d', 'func')]),
    (datasource, z_score, [('brainmask','mask')]),
    (inputnode, z_score, [('subj_id', 'subj_id'),
                         ('run_id', 'run_id')]),
    (z_score, concatenate, [('res_file_demean','in_files')]),
    (inputnode, rename_merge, [('subj_id','subj_id')]),
    (concatenate, rename_merge, [('merged_file', 'in_file')]),
    (rename_merge, datasink, [('out_file','before_FIR')])
])

# -----------
# FIR-GLM
# -----------

# create a shared brain mask for all 6 runs
shared_mask = pe.Node(
    interface = niu.Function(
        input_names = ['mask_list','subj_id'],
        output_names = ['shared_mask_img_path'],
        function = create_shared_mask),
    name = 'shared_mask')


# fsl.GLM
fir_glm = pe.Node(
    interface = fsl.GLM(),
    name = 'fir_glm'
)
fir_glm.inputs.output_type = 'NIFTI_GZ'
fir_glm.inputs.out_file = os.path.join(os.getcwd(),'beta_param.nii.gz')
fir_glm.inputs.out_res_name = os.path.join(os.getcwd(),'res4d.nii.gz')

# beta image output file name
beta_image = pe.Node(
    interface = niu.Function(
        input_names = ['subj_id'],
        output_names = ['beta_file_name'],
        function = _beta_naming),
    name = 'beta_image'
    
)

# beta image output file name
residual_image = pe.Node(
    interface = niu.Function(
        input_names = ['subj_id'],
        output_names = ['residual_file_name'],
        function = _res_naming),
    name = 'residual_image'
)

wf.connect([
    (inputnode, shared_mask, [('subj_id', 'subj_id')]),
    (datasource, shared_mask, [('brainmask', 'mask_list')]),
    (rename_merge, fir_glm, [('out_file', 'in_file')]),
    (designsource, fir_glm, [('design_mat', 'design')]),
    (shared_mask, fir_glm, [('shared_mask_img_path','mask')]),
    
    (inputnode, beta_image, [('subj_id', 'subj_id')]),
    (beta_image, fir_glm, [('beta_file_name', 'out_file')]),
    (inputnode, residual_image, [('subj_id','subj_id')]),
    (residual_image, fir_glm, [('residual_file_name', 'out_res_name')]),
    (fir_glm, datasink, [('out_file', 'FIR_betas')]),
    (fir_glm, datasink, [('out_res', 'FIR_residual')])
])

# ----------------
# Run the workflow
# ----------------

wf.write_graph(graph2use='colored')
wf.config['logging'] = {'log_to_file': 'true', 'log_directory': log_dir}
wf.config['execution'] = {
    'stop_on_first_crash': 'true',
    'crashfile_format': 'txt',
    'crashdump_dir': log_dir,
    'job_finished_timeout': '65'
}
wf.config['monitoring'] = {'enabled': 'true'}
wf.run(plugin='MultiProc', plugin_args={'n_procs': n_proc})