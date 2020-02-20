#!/bin/bash

# Running this dataprep recipe does the following:

# Makes Kaldi data folder for VoxCeleb 2 (just train portion)
# Makes Kaldi data folder for VoxCeleb 1 (train+test portion)
# Makes MFCCs for each dataset
# Augments the VoxCeleb 2 train portion with MUSAN and RIR_NOISES data, in addition to removing silence frames.
# Removes silence frames from VoxCeleb 1
# The Kaldi data folder $KALDI_ROOT/egs/voxceleb/v2/data/train_combined_no_sil is the end result. If done correctly, the resulting train dataset should have 5994 speakers (5994 lines in spk2utt)

#cp scripts/run_vc_dataprep.sh $KALDI_ROOT/egs/voxceleb/v2/
#cd $KALDI_ROOT/egs/voxceleb/v2
#rm -rf data/
#rm -rf exp/
#source path.sh
#./run_vc_dataprep.sh


# Similar to the VoxCeleb data prep

#cp scripts/run_sitw_dataprep.sh $KALDI_ROOT/egs/sitw/v2/
#cd $KALDI_ROOT/egs/sitw/v2
#rm -rf data/
#rm -rf exp/
#source path.sh
#./run_sitw_dataprep.sh


#Additional necessary data prep

#For speaker datasets intended to be used as evaluation/test datasets, there must also be a file called veri_pairs within these data folders.
#This is similar to a trials file used by Kaldi which lists the pairs of utterances that are to be compared, along with the true label of whether or not they belong to the same speaker.

#The format of this veri_pairs file is as follows:

# 1 <utterance_a> <utterance_b>
# 0 <utterance_a> <utterance_c>


# To obtain the primary verification list for VoxCeleb, the following code can be run:

#wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt
#python scripts/vctxt_to_veripairs.py veri_test.txt $KALDI_ROOT/egs/voxceleb/v2/data/voxceleb1_nosil/veri_pairs

# For SITW, the dev and eval core-core lists are merged into a kaldi-like data folder for each, which contains the features and veri_pairs file needed for evaluation. This is performed from the repository root like so.

#python scripts/trials_to_veri_pairs.py $KALDI_ROOT/egs/sitw/v2/data