# -*- coding: utf-8 -*-

"""
post fmriprep - smoothing & high pass
"""

import os
import argparse
from nipype import Node, Workflow
from nipype.pipeline import engine as pe
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
# Used to be under nipype.workflows.fmri.fsl, now they made it into a new module. 
import niflow.nipype1.workflows.fmri.fsl as fsl_wf

parser = argparse.ArgumentParser(description='Post FMRIPREP preprocess pipeline.')
parser.add_argument(
    '--base_dir', 
    action='store', 
    default='/projects/hulacon/shared/strongbad01/strongbad01/subjects',
    help='The directory where Derivative exist.')
parser.add_argument(
    '--task-id',
    action='store',
    default = 'SB',
    help='Select a specific task to be processed.')
parser.add_argument(
    '--bold-space',
    action='store',
    help='The space EPI data normalized into.',
    default = 'MNI152NLin2009cAsym_res-native')
parser.add_argument(
    '--sub-id',
    required=True,
    action='store',
    help='subject id')
parser.add_argument(
    '--smooth-fwhm',
    action='store',
    type=float,
    default=5,
    help='Susan smooth FWHM (in mm).')
parser.add_argument(
    '--highpass',
    action='store',
    type=float,
    default=50,
    help='Sigma in volumn (filter cut off in seconds / 2*TR).')
parser.add_argument(
    '--n_procs',
    action='store',
    default=1,
    type=int,
    help='Maximum number of threads across all processes.')



args = parser.parse_args()

# -----------------------
# Setup basic inforamtion
# -----------------------
# Directories information
base_dir = args.base_dir # the directory where "derivative" exists (derivative from fmriprep)
output_dir = os.path.join(base_dir,'post_fmriprep_check') # the output directory, I made it to be parallel with the derivative folder 
                                                 # importantly, within this directory, make an empty working directory.
deriv_dir = os.path.join(base_dir,'derivative') 
src_dir = os.path.join(deriv_dir,'fmriprep') 

# Subject, Run, Task information
# The template for datagrabber below, change input based on the template. 
## func = 'fmriprep/sub-%s/func/sub-%s_task-%s_run-%s_space-%s_desc-preproc_bold.nii.gz'
subj_id = args.sub_id
taskid = args.task_id
bold_space = args.bold_space
run_id = ['{:02d}'.format(x) for x in range(1,7)]
fwhm = args.smooth_fwhm 
hpcutoff = args.highpass 
n_proc = args.n_procs

# ----------------------
# Preprocess pipeline
# ----------------------
    
# start a wf for each subject
os.mkdir(os.path.join(output_dir,'working',subj_id))
wf = pe.Workflow(name = 'datainput')
wf.base_dir = os.path.join(output_dir,'working',subj_id)

# Feed the dynamic parameters into the node
inputnode = pe.Node(
    interface = niu.IdentityInterface(
    fields = ['subj_id','task_id','run_id','bold_space','brainmask','fwhm']), name = 'inputnode')
# specified above
inputnode.inputs.subj_id = subj_id
inputnode.inputs.task_id = taskid
inputnode.inputs.bold_space = bold_space
inputnode.inputs.fwhm = fwhm
inputnode.iterables = [('run_id',run_id)]

# grab input data
datasource = pe.Node(
    interface = nio.DataGrabber(
        infields = ['subj_id','task_id','run_id','bold_space'],
        outfields = ['func','brainmask']),
        name = 'datasource')
# Location of the dataset folder
datasource.inputs.base_directory = deriv_dir    
# Necessary default parameters 
datasource.inputs.template = '*' 
datasource.inputs.sort_filelist = True
# the string template to match
# field_template and template_args are both dictionaries whose keys correspond to the outfields keyword
datasource.inputs.template_args = dict(
    func=[['subj_id', 'subj_id', 'task_id', 'run_id', 'bold_space']],
    brainmask =[['subj_id', 'subj_id', 'task_id', 'run_id', 'bold_space']])
datasource.inputs.field_template = dict(
    func = 'fmriprep/sub-%s/func/sub-%s_task-%s_run-%s_space-%s_desc-preproc_bold.nii.gz',
    brainmask = 'fmriprep/sub-%s/func/sub-%s_task-%s_run-%s_space-%s_desc-brain_mask.nii.gz')

# connect the input node with datagrabber, so the datagrabber will use the dynamic parameters to grab func and brain mask for each run
wf.connect([(inputnode, datasource, [('subj_id', 'subj_id'),
                                        ('task_id', 'task_id'),
                                        ('run_id', 'run_id'),
                                        ('bold_space', 'bold_space'),
                                        ('brainmask', 'brainmask')])])
                    
# specify a data sinker
datasink = pe.Node(interface = nio.DataSink(), name = 'datasink')
datasink.inputs.base_directory = output_dir
datasink.inputs.parameterization = False # the output structure do not contain iterable folder (wont have run-01/2/3 folder)
datasink.inputs.substitutions = [('_smooth',''),
                                ('_tempfilt','')]

# Use create_susan_smooth, a workflow from Niflow for smoothing
smooth = fsl_wf.create_susan_smooth()
select_smoothed = pe.Node(interface=niu.Select(), name='SelectSmoothed')
select_smoothed.inputs.index = 0 # output a list of lots of files, only choose smoothing related file 

# connect to the workflow
wf.connect([(inputnode, smooth, [('fwhm', 'inputnode.fwhm')]),
                (datasource, smooth, [('func','inputnode.in_files')]),
                (datasource, smooth, [('brainmask', 'inputnode.mask_file')]),
                (smooth, select_smoothed, [('outputnode.smoothed_files', 'inlist')])])
            # (select_smoothed, datasink, [('out','smooth1.@smooth1_file')])])


# temporal mean and high pass node
# fsl highpass will demean, here create a mean image for latter added back
temporal_mean = pe.Node(interface = fsl.MeanImage(), name = 'temporal_mean') 

# bandpass-temporal filtering, two inputs: hp sigma and lp sigma, 
# unit is sigma in volumn not in seconds 
# set low pass sigma to -1 to skip this filter
# hp sigma in volumn = FWHM / 2*TR \
temporal_filter = pe.Node(interface = fsl.ImageMaths(suffix = '_tempfilt'), name = 'temporal_filter')
temporal_filter.inputs.op_string = (f'-bptf {hpcutoff} -1 -add')

# connect to the workflow
wf.connect([(select_smoothed, temporal_mean,[('out','in_file')]),
            (select_smoothed, temporal_filter, [('out','in_file')]),
            (temporal_mean, temporal_filter, [('out_file','in_file2')]),
            (temporal_filter, datasink, [('out_file',(f'sub-{subj_id}.@preprocessed'))])])

# Run preproc workflow
wf.write_graph(graph2use='colored')
wf.config['logging'] = {'log_to_file': 'true', 'log_directory': output_dir}
wf.config['execution'] = {
    'stop_on_first_crash': 'true',
    'crashfile_format': 'txt',
    'crashdump_dir': output_dir,
    'job_finished_timeout': '65'
}
wf.config['monitoring'] = {'enabled': 'true'}
wf.run(plugin='MultiProc', plugin_args={'n_procs': n_proc})
    
