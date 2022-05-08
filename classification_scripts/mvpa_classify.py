#  Copyright 2016 Intel Corporation
##
## Author: Y.Peeta Li; modified based on original scripts from BrainIAK https://github.com/brainiak/brainiak/tree/master/brainiak/fcma
## Email: peetal@uoregon.edu
##

from brainiak.fcma.preprocessing import prepare_mvpa_data
from brainiak.fcma.preprocessing import prepare_searchlight_mvpa_data
from brainiak import io

from sklearn.svm import SVC
import sys
import logging
import numpy as np
from sklearn import model_selection
from mpi4py import MPI
import os
from brainiak.image import mask_images
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
output_file = results_path + '/classify_result_decision_making_func.txt'
output_csv = results_path + f"/sub{left_out_subj}_classify_result.csv"

# Do you want to compute this in an easily understood way (0) or a memory efficient way (1)?
is_memory_efficient = 1

# If a second mask was supplied then this is an extrinsic analysis and treat it as such
if second_mask == "None":
    is_extrinsic = 0
else:
    is_extrinsic = 1  
 
if __name__ == '__main__':
    
    # Send a message on the first node
    if MPI.COMM_WORLD.Get_rank()==0:
        logger.info(
            'Testing for participant %d.\nProgramming starts in %d process(es)' %
            (int(left_out_subj), MPI.COMM_WORLD.Get_size())
        )
    
    # Load in the volumes, mask and labels
    images = io.load_images_from_dir(data_dir, suffix=suffix)
    top_n_mask = io.load_boolean_mask(top_n_mask_file)
    epoch_list = io.load_labels(epoch_file)

    # Parse the epoch data for useful dimensions
    # epochs_per_subj = epoch_list[0].shape[1] # comment out b/c im doing 2 out 3 ways classification, 
        # only care about two out of three conditions, 32 out of 48 epochs. 
    epochs_per_subj = 32
    num_subjs = len(epoch_list)
    
    # Prepare the data
    int_data, labels = prepare_mvpa_data(images, epoch_list, top_n_mask)

    # What indexes pick out the left out participant?
    start_idx = int(int(left_out_subj) * epochs_per_subj)
    end_idx = int(start_idx + epochs_per_subj)
    
    # Take out the idxs corresponding to all participants but this one
    training_idx = list(set(range(len(labels))) - set(range(start_idx, end_idx)))
    testing_idx = list(range(start_idx, end_idx))
    
    # Pull out the data
    int_data_training = [int_data[:,i] for i in training_idx]
    int_data_testing = [int_data[:,i] for i in testing_idx]
    
    # Pull out the labels
    labels_training = [labels[i] for i in training_idx]
    labels_testing = [labels[i] for i in testing_idx]
    
    # create model
    svm_clf = SVC(kernel='linear', C=1)

    # train the model
    svm_clf.fit(int_data_training, labels_training)

    # test the model
    predict = svm_clf.predict(int_data_testing)
    corr = np.sum(predict == labels_testing)
    score = svm_clf.score(int_data_testing, labels_testing)
    decision_function = svm_clf.decision_function(int_data_testing)

    # Report results on the first rank core
    if MPI.COMM_WORLD.Get_rank()==0:
        # print('--RESULTS--')
        
        # intrinsic_vs_extrinsic = ['intrinsic', 'extrinsic']
        
        # # Report accuracy
        # logger.info(
        #     'When leaving subject %d out for testing using the %s mask for an %s correlation, the accuracy is %d / %d = %.2f' %
        #     (int(left_out_subj), top_n_mask_file, intrinsic_vs_extrinsic[int(is_extrinsic)], corr, epochs_per_subj, score)
        # )
        
        # # Append this accuracy on to a score sheet
        # with open(output_file, 'a') as fp:
        #     fp.write(top_n_mask_file + ', ' + str(intrinsic_vs_extrinsic[int(is_extrinsic)]) + ': ' + str(score) + '\n')  
        #     fp.write(top_n_mask_file + ',: ' + str(decision_function) + '\n')
        df = pd.DataFrame({"sub_id": np.repeat(int(left_out_subj), 32).tolist(), 
                            "acc": np.repeat(corr, 32).tolist(), 
                            "conf": decision_function.tolist(),
                            "cond": np.repeat(str(cond), 32).tolist(),
                            })
        df.to_csv(output_csv, index = False)
