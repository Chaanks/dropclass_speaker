#!/bin/bash

#Running this dataprep recipe does the following:

#Makes Kaldi data folder for VoxCeleb 2 (just train portion)
#Makes Kaldi data folder for VoxCeleb 1 (train+test portion)
#Makes MFCCs for each dataset
#Augments the VoxCeleb 2 train portion with MUSAN and RIR_NOISES data, in addition to removing silence frames.
#Removes silence frames from VoxCeleb 1
#The Kaldi data folder $KALDI_ROOT/egs/voxceleb/v2/data/train_combined_no_sil is the end result. If done correctly, the resulting train dataset should have 5994 speakers (5994 lines in spk2utt)

cp scripts/run_vc_dataprep.sh $KALDI_ROOT/egs/voxceleb/v2/
cd $KALDI_ROOT/egs/voxceleb/v2
rm -rf data/
rm -rf exp/
source path.sh
./run_vc_dataprep.sh


#Similar to the VoxCeleb data prep

#cp scripts/run_sitw_dataprep.sh $KALDI_ROOT/egs/sitw/v2/
#cd $KALDI_ROOT/egs/sitw/v2
#source path.sh
#./run_sitw_dataprep.sh
