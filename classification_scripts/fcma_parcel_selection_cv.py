#  Copyright 2016 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from brainiak.fcma.voxelselector import VoxelSelector
from brainiak.fcma.preprocessing import RandomType
from brainiak import io
from sklearn.svm import SVC
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
from nilearn import image
from mpi4py import MPI
import logging
import numpy as np
import nibabel as nib
import os
import time
import math
from scipy.stats.mstats import zscore
from enum import Enum
import copy

def _parcel_mask_images(images, mask, data_type: type = None
                       ) -> np.ndarray:
    """ mask out data using parcellation scheme. 

    Parameters
    ----------
    images: iterable of Niimg-like object
    mask: Niimg-like object

    Returns
    -------
    activity_data1: a list of 2D array. The length of the list is the number of subject, 
        the shape of the 2D array is nparcel x nTR. 

    """
    # generate seperate masks
    parcel_id = np.unique(mask.get_fdata())[1:]
    activity_data1 = []

    for _,image in enumerate(images):
        ts_parcel_per_sub = []
        mask_dat = mask.get_fdata()
        image_dat = image.get_fdata()
        for pid in parcel_id:
            mask_dat_copy = mask_dat.copy()
            index_dump = np.where(mask_dat != pid)
            mask_dat_copy[index_dump] = 0
            per_parcel_mask = mask_dat_copy.astype(np.bool)

            # mask out the voxels within a parcel
            per_parcel_data = image_dat[per_parcel_mask]

            # get average beta estimate for all voxels within that parcel
            per_parcel_data[per_parcel_data == 0] = np.nan
            parcel_mean_estimate = np.nanmean(per_parcel_data, axis = 0)

            # append to parcel_timeseries
            ts_parcel_per_sub.append(parcel_mean_estimate)
        
        activity_data1.append(np.stack(ts_parcel_per_sub, axis = 0).astype(data_type))
    
    return(activity_data1)

def _separate_epochs(activity_data, epoch_list):
    """ create data epoch by epoch

    Separate data into epochs of interest specified in epoch_list
    and z-score them for computing correlation

    Parameters
    ----------
    activity_data: list of 2D array in shape [nVoxels, nTRs]
        the masked activity data organized in voxel*TR formats of all subjects
    epoch_list: list of 3D array in shape [condition, nEpochs, nTRs]
        specification of epochs and conditions
        assuming all subjects have the same number of epochs
        len(epoch_list) equals the number of subjects

    Returns
    -------
    raw_data: list of 2D array in shape [epoch length, nVoxels]
        the data organized in epochs
        and z-scored in preparation of correlation computation
        len(raw_data) equals the number of epochs
    labels: list of 1D array
        the condition labels of the epochs
        len(labels) labels equals the number of epochs
    """
    time1 = time.time()
    raw_data = []
    labels = []
    for sid in range(len(epoch_list)):
        epoch = epoch_list[sid]
        for cond in range(epoch.shape[0]):
            sub_epoch = epoch[cond, :, :]
            for eid in range(epoch.shape[1]):
                r = np.sum(sub_epoch[eid, :])
                if r > 0:   # there is an epoch in this condition
                    # mat is row-major
                    # regardless of the order of acitvity_data[sid]
                    mat = activity_data[sid][:, sub_epoch[eid, :] == 1]
                    mat = np.ascontiguousarray(mat.T)
                    mat = zscore(mat, axis=0, ddof=0)
                    # if zscore fails (standard deviation is zero),
                    # set all values to be zero
                    mat = np.nan_to_num(mat)
                    mat = mat / math.sqrt(r)
                    raw_data.append(mat)
                    labels.append(cond)
    time2 = time.time()
    logger.debug(
        'epoch separation done, takes %.2f s' %
        (time2 - time1)
    )
    return raw_data, labels


def _prepare_fcma_parcel_data(images, conditions, mask1, mask2=None,
                       comm=MPI.COMM_WORLD):
    """Prepare data for correlation-based computation and analysis.

    Generate epochs of interests, then broadcast to all workers.

    Parameters
    ----------
    images: Iterable[SpatialImage]
        Data.
    conditions: List[UniqueLabelConditionSpec]
        Condition specification.
    mask1: np.ndarray
        Mask to apply to each image.
    mask2: Optional[np.ndarray]
        Mask to apply to each image.
        If it is not specified, the method will assign None to the returning
        variable raw_data2 and the self-correlation on raw_data1 will be
        computed
    random: Optional[RandomType]
        Randomize the image data within subject or not.
    comm: MPI.Comm
        MPI communicator to use for MPI operations.

    Returns
    -------
    raw_data1: list of 2D array in shape [epoch length, nVoxels]
        the data organized in epochs, specified by the first mask.
        len(raw_data) equals the number of epochs
    raw_data2: Optional, list of 2D array in shape [epoch length, nVoxels]
        the data organized in epochs, specified by the second mask if any.
        len(raw_data2) equals the number of epochs
    labels: list of 1D array
        the condition labels of the epochs
        len(labels) labels equals the number of epochs
    """
    rank = comm.Get_rank()
    labels = []
    raw_data1 = []
    raw_data2 = []
    if rank == 0:
        logger.info('start to apply masks and separate epochs')
        activity_data1 = _parcel_mask_images(images, mask1, np.float32)
        raw_data1, labels = _separate_epochs(activity_data1, conditions)
        time1 = time.time()
    raw_data_length = len(raw_data1)
    raw_data_length = comm.bcast(raw_data_length)

    # broadcast the data subject by subject to prevent size overflow
    for i in range(raw_data_length):
        if rank != 0:
            raw_data1.append(None)
            if mask2 is not None:
                raw_data2.append(None)
        raw_data1[i] = comm.bcast(raw_data1[i], root=0)
        if mask2 is not None:
            raw_data2[i] = comm.bcast(raw_data2[i], root=0)

    if comm.Get_size() > 1:
        labels = comm.bcast(labels, root=0)
        if rank == 0:
            time2 = time.time()
            logger.info(
                'data broadcasting done, takes %.2f s' %
                (time2 - time1)
            )
    if mask2 is None:
        raw_data2 = None
    return raw_data1, raw_data2, labels



format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)

"""
Perform leave one participant out voxel selection with FCMA
"""

data_dir = sys.argv[1]  # What is the directory containing data?
suffix = sys.argv[2]  # What is the extension of the data you're loading
mask_file = sys.argv[3]  # What is the path to the whole brain mask
epoch_file = sys.argv[4]  # What is the path to the epoch file
left_out_subj = sys.argv[5]  # Which participant (as an integer) are you leaving out for this cv?
output_dir = sys.argv[6]  # What is the path to the folder you want to save this data in

# Only run the following from the controller core
if __name__ == '__main__':
    if MPI.COMM_WORLD.Get_rank()==0:
        logger.info(
            'Testing for participant %d.\nProgramming starts in %d process(es)' %
            (int(left_out_subj), MPI.COMM_WORLD.Get_size())
        )
        # create output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    # Load in the volumes, mask and labels
    images = io.load_images_from_dir(data_dir, suffix=suffix)
    mask = image.load_img(mask_file)
    epoch_list = io.load_labels(epoch_file)

    # Parse the epoch data for useful dimensions
    epochs_per_subj = 32
    num_subjs = len(epoch_list)
    
    # Preprocess the data and prepare for FCMA
    raw_data, _, labels = _prepare_fcma_parcel_data(images, epoch_list, mask)

    # enforce left one out
    file_str = output_dir + '/fc_no' + str(left_out_subj) + '_'
    start_idx = int(int(left_out_subj) * epochs_per_subj)
    end_idx = int(start_idx + epochs_per_subj)
    
    # Take out the idxs corresponding to all participants but this one
    subsampled_idx = list(set(range(len(labels))) - set(range(start_idx, end_idx)))
    labels_subsampled = [labels[i] for i in subsampled_idx]
    raw_data_subsampled = [raw_data[i] for i in subsampled_idx]
    
    # Set up the voxel selection object for fcma
    vs = VoxelSelector(labels_subsampled, epochs_per_subj, num_subjs - 1, raw_data_subsampled)
    
    # for cross validation, use SVM with precomputed kernel
    clf = SVC(kernel='precomputed', shrinking=False, C=1)
    results = vs.run(clf)
    
    # this output is just for result checking
    if MPI.COMM_WORLD.Get_rank()==0:
        logger.info(
            'correlation-based voxel selection is done'
        )
        
        # Write a text document of the voxel selection results
        #with open(file_str + 'result_list.txt', 'w') as fp:
        #    for idx, tuple in enumerate(results):
        #        fp.write(str(tuple[0]) + ' ' + str(tuple[1]) + '\n')
        
        # Load in the mask with nibabel
        mask_img = nib.load(mask_file)

        # Preset the volumes
        score_volume = copy.deepcopy(mask_img.get_fdata())
        score = np.zeros(len(results), dtype=np.float32)

        seq_volume = copy.deepcopy(mask_img.get_fdata())
        seq = np.zeros(len(results), dtype=np.int)

        # Write a text document of the voxel selection results
        with open(file_str + 'result_list.txt', 'w') as fp:
            for idx, tuple in enumerate(results):
                fp.write(str(tuple[0]) + ' ' + str(tuple[1]) + '\n')

                # Store the score for each voxel
                score[tuple[0]] = tuple[1]
                seq[tuple[0]] = idx

        # Convert the list into a volume
        for idx, acc_score in enumerate(score):
            parcel_id = idx + 1
            score_volume[np.where(score_volume == parcel_id)] = acc_score

        for idx, acc_seq in enumerate(seq):
            parcel_id = idx + 1
            seq_volume[np.where(seq_volume == parcel_id)] = acc_seq



        # Save volume
        io.save_as_nifti_file(score_volume, mask_img.affine,
                                file_str + 'result_score.nii.gz')
        io.save_as_nifti_file(seq_volume, mask_img.affine,
                                file_str + 'result_seq.nii.gz')



