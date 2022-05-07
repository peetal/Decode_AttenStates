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

from brainiak.fcma.classifier import Classifier
from brainiak.fcma.preprocessing import RandomType
from brainiak import io
import sys
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
from nilearn import image
from sklearn.svm import SVC
import logging
import numpy as np
from sklearn import model_selection
from mpi4py import MPI
import os
import nibabel as nib
import time
import math
from scipy.stats.mstats import zscore
from enum import Enum
import pandas as pd

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)

data_dir = sys.argv[1]
suffix = sys.argv[2]
top_n_mask_file = sys.argv[3] # This is not the whole brain mask! This is the voxel selection mask
epoch_file = sys.argv[4]
left_out_subj = sys.argv[5]
results_path = sys.argv[6]
cond = sys.argv[7]
if len(sys.argv)==9:
    second_mask = sys.argv[8]  # Do you want to supply a second mask (for extrinsic analysis)
else:
    second_mask = "None"
    
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Where do you want to output the classification results?
output_file = results_path + '/classify_result.txt'
output_csv = results_path + f"/sub{left_out_subj}_classify_result.csv"

# Do you want to compute this in an easily understood way (0) or a memory efficient way (1)?
is_memory_efficient = 1

# If a second mask was supplied then this is an extrinsic analysis and treat it as such
if second_mask == "None":
    is_extrinsic = 0
else:
    is_extrinsic = 1  
    
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



if __name__ == '__main__':
    
    # Send a message on the first node
    if MPI.COMM_WORLD.Get_rank()==0:
        logger.info(
            'Testing for participant %d.\nProgramming starts in %d process(es)' %
            (int(left_out_subj), MPI.COMM_WORLD.Get_size())
        )
    
    # Load in the volumes, mask and labels
    images = io.load_images_from_dir(data_dir, suffix=suffix)
    top_n_mask = image.load_img(top_n_mask_file)
    epoch_list = io.load_labels(epoch_file)

    # Parse the epoch data for useful dimensions
    # epochs_per_subj = epoch_list[0].shape[1] # comment out b/c im doing 2 out 3 ways classification, 
        # only care about two out of three conditions, 32 out of 48 epochs. 
    epochs_per_subj = 32
    num_subjs = len(epoch_list)
    
    # Prepare the data
    int_data, _, labels = _prepare_fcma_parcel_data(images, epoch_list, top_n_mask)
    
    # What indexes pick out the left out participant?
    start_idx = int(int(left_out_subj) * epochs_per_subj)
    end_idx = int(start_idx + epochs_per_subj)
    
    # Take out the idxs corresponding to all participants but this one
    training_idx = list(set(range(len(labels))) - set(range(start_idx, end_idx)))
    testing_idx = list(range(start_idx, end_idx))
    
    # Pull out the data
    int_data_training = [int_data[i] for i in training_idx]
    int_data_testing = [int_data[i] for i in testing_idx]
    
    # Pull out the labels
    labels_training = [labels[i] for i in training_idx]
    labels_testing = [labels[i] for i in testing_idx]
    
    # Prepare the data to be processed efficiently (albeit in a less easy to follow way)
    if is_memory_efficient == 1:
        rearranged_int_data = int_data_training + int_data_testing
        rearranged_labels = labels_training + labels_testing
        num_training_samples = epochs_per_subj * (num_subjs - 1)
    
    # Do you want to perform an intrinsic vs extrinsic analysis
    if is_extrinsic > 0 and is_memory_efficient == 1:
        
        # This needs to be reloaded every time you call prepare_fcma_data
        images = io.load_images_from_dir(data_dir, suffix=suffix)
        
        # Multiply the inverse of the top n mask by the whole brain mask to bound it
        second_mask = io.load_boolean_mask(second_mask)
        extrinsic_mask = ((top_n_mask == 0) * second_mask)==1
        
        # Prepare the data using the extrinsic data
        ext_data, _, _ = prepare_fcma_data(images, epoch_list, extrinsic_mask)

        # Pull out the appropriate extrinsic data
        ext_data_training = [ext_data[i] for i in training_idx]
        ext_data_testing = [ext_data[i] for i in testing_idx]

        # Set up data so that the internal mask is correlated with the extrinsic mask
        rearranged_ext_data = ext_data_training + ext_data_testing
        corr_obj = list(zip(rearranged_ext_data, rearranged_int_data))
    else:
        
        # Set up data so that the internal mask is correlated with the internal mask
        if is_memory_efficient == 1:
            corr_obj = list(zip(rearranged_int_data, rearranged_int_data))
        else:
            training_obj = list(zip(int_data_training, int_data_training))
            testing_obj = list(zip(int_data_testing, int_data_testing))
    
    # no shrinking, set C=1
    svm_clf = SVC(kernel='precomputed', shrinking=False, C=1)
    
    clf = Classifier(svm_clf, epochs_per_subj=epochs_per_subj)
    
    # Train the model on the training data
    if is_memory_efficient == 1:
        clf.fit(corr_obj, rearranged_labels, num_training_samples)
    else:
        clf.fit(training_obj, labels_training)
    
    # What is the cv accuracy?
    if is_memory_efficient == 0:
        cv_prediction = clf.predict(training_obj)
    
    # Test on the testing data
    if is_memory_efficient == 1:
        predict = clf.predict()
    else:
        predict = clf.predict(testing_obj)

    # Report results on the first rank core
    if MPI.COMM_WORLD.Get_rank()==0:
        print('--RESULTS--')
        print(clf.decision_function())
        
        # How often does the prediction match the target
        num_correct = (np.asanyarray(predict) == np.asanyarray(labels_testing)).sum()
        
        # Print the CV accuracy
        if is_memory_efficient == 0:
            cv_accuracy = (np.asanyarray(cv_prediction) == np.asanyarray(labels_training)).sum() / len(labels_training)
            print('CV accuracy: %0.5f' % (cv_accuracy))
        
        intrinsic_vs_extrinsic = ['intrinsic', 'extrinsic']
        
        # Report accuracy
        logger.info(
            'When leaving subject %d out for testing using the %s mask for an %s correlation, the accuracy is %d / %d = %.2f' %
            (int(left_out_subj), top_n_mask_file, intrinsic_vs_extrinsic[int(is_extrinsic)], num_correct, epochs_per_subj, num_correct / epochs_per_subj)
        )
        
        # Append this accuracy on to a score sheet
        with open(output_file, 'a') as fp:
            fp.write(top_n_mask_file + ', ' + str(intrinsic_vs_extrinsic[int(is_extrinsic)]) + ': ' + str(num_correct / epochs_per_subj) + '\n')    
        
        
        df = pd.DataFrame({"sub_id": np.repeat(int(left_out_subj), 32).tolist(), 
                            "acc": np.repeat(round(num_correct / epochs_per_subj,2), 32).tolist(), 
                            "conf": clf.decision_function().tolist(),
                            "cond": np.repeat(str(cond), 32).tolist()})
        df.to_csv(output_csv, index = False)