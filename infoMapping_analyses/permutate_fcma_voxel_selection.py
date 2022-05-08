from brainiak.fcma.voxelselector import VoxelSelector
from brainiak.fcma.preprocessing import prepare_fcma_data
from brainiak.fcma.preprocessing import RandomType
from brainiak import io
from sklearn.svm import SVC
import sys
from mpi4py import MPI
import logging
import numpy as np
import nibabel as nib
import os
from glob import glob

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
permutate_epoch_dir = sys.argv[4] # The directory of the permutated epoch files. 
start_epochfile_id = sys.argv[5] # int indicating which permutated epoch file to use. 
iteration = sys.argv[6]
output_dir = sys.argv[7]  # What is the path to the folder you want to save this data in

# grab all the epoch files
epoch_file_list = sorted(glob(os.path.join(permutate_epoch_dir, '*.npy')), key= lambda x: int(x.split('_')[-1][3:-4]))
# select the epoch file that will be used
selected_epoch = epoch_file_list[int(start_epochfile_id) : int(start_epochfile_id)+int(iteration)]


if __name__ == '__main__':

    # Only run the following from the controller core
    for i in range(int(iteration)):
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.info(
                'Testing for iteration %d.\nProgramming starts in %d process(es)' %
                (int(i), MPI.COMM_WORLD.Get_size())
            )
            # create output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        # Load in the volumes, mask and labels
        images = io.load_images_from_dir(data_dir, suffix=suffix)
        mask_whole_brain = io.load_boolean_mask(mask_file)
        logger.info(f"The current epoch file is {selected_epoch[i]}")
        epoch_list = io.load_labels(selected_epoch[i])

            
        # Parse the epoch data for useful dimensions
        epochs_per_subj = 32
        num_subjs = len(epoch_list)

        # Preprocess the data and prepare for FCMA
        raw_data, _ , labels = prepare_fcma_data(images, epoch_list, mask_whole_brain)

        # Do not do leave one out
        iterate_id = int(start_epochfile_id) + int(i)
        file_str = output_dir + '/fc_permutate_label_shuffled' + str(iterate_id) + '_'

        # Set up the voxel selection object for fcma
        vs = VoxelSelector(labels, epochs_per_subj, num_subjs, raw_data)

        # for cross validation, use SVM with precomputed kernel
        clf = SVC(kernel='precomputed', shrinking=False, C=1)
        results = vs.run(clf)

        # this output is just for result checking
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.info(
                'correlation-based voxel selection is done'
            )

            # Load in the mask with nibabel
            mask_img = nib.load(mask_file)
            mask = mask_img.get_data().astype(np.bool)

            # Preset the volumes
            score_volume = np.zeros(mask.shape, dtype=np.float32)
            score = np.zeros(len(results), dtype=np.float32)
            seq_volume = np.zeros(mask.shape, dtype=np.int)
            seq = np.zeros(len(results), dtype=np.int)

            # Write a text document of the voxel selection results
            with open(file_str + 'result_list.txt', 'w') as fp:
                for idx, tuple in enumerate(results):
                    fp.write(str(tuple[0]) + ' ' + str(tuple[1]) + '\n')

                    # Store the score for each voxel
                    score[tuple[0]] = tuple[1]
                    seq[tuple[0]] = idx

            # Convert the list into a volume
            score_volume[mask] = score
            seq_volume[mask] = seq

            # Save volume
            io.save_as_nifti_file(score_volume, mask_img.affine,
                                    file_str + 'result_score.nii.gz')
            io.save_as_nifti_file(seq_volume, mask_img.affine,
                                    file_str + 'result_seq.nii.gz')

