. ./cmd.sh
. ./path.sh
set -e

nnet_dir=exp/$1
stage=$2

voxceleb1_root=/data/egs/voxceleb/v2
voxceleb1_trials=$voxceleb1_root/data/voxceleb1_nosil/trials

if [ $stage -le 0 ]; then
  # Compute the mean.vec used for centering.
  $train_cmd $nnet_dir/xvectors_train_combined_200k/log/compute_mean.log \
    ivector-mean scp:$nnet_dir/xvectors_train_combined_200k/xvector.scp \
    $nnet_dir/xvectors_train_combined_200k/mean.vec || exit 1;

  # Use LDA to decrease the dimensionality prior to PLDA.
  lda_dim=128
  $train_cmd $nnet_dir/xvectors_train_combined_200k/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train_combined_200k/xvector.scp ark:- |" \
    ark:$voxceleb1_root/data/train_combined_200k/utt2spk $nnet_dir/xvectors_train_combined_200k/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnet_dir/xvectors_train_combined_200k/log/plda.log \
    ivector-compute-plda ark:$voxceleb1_root/data/train_combined_200k/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train_combined_200k/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnet_dir/xvectors_train_combined_200k/plda || exit 1;
fi

if [ $stage -le 1 ]; then
  $train_cmd $nnet_dir/scores/log/voxceleb1_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train_combined_200k/plda - |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train_combined_200k/mean.vec scp:$nnet_dir/xvectors_voxceleb1/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train_combined_200k/mean.vec scp:$nnet_dir/xvectors_voxceleb1/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $nnet_dir/scores/scores_voxceleb1_test || exit 1;

  echo "Voxceleb1:"
  eer=$(paste $voxceleb1_trials $nnet_dir/scores/scores_voxceleb1_test | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`scripts/compute_min_dcf.py --p-target 0.01 $nnet_dir/scores/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  mindcf2=`scripts/compute_min_dcf.py --p-target 0.001 $nnet_dir/scores/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi

## TODO SITW SCORING
#if [ $stage -le 12 ]; then
#  # Compute PLDA scores for SITW dev core-core trials
#  $train_cmd $nnet_dir/scores/log/sitw_dev_core_scoring.log \
#    ivector-plda-scoring --normalize-length=true \
#    --num-utts=ark:$nnet_dir/xvectors_sitw_dev_enroll/num_utts.ark \
#    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train_combined_200k/plda - |" \
#    "ark:ivector-mean ark:data/sitw_dev_enroll/spk2utt scp:$nnet_dir/xvectors_sitw_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnet_dir/xvectors_train_combined_200k/mean.vec ark:- ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train_combined_200k/mean.vec scp:$nnet_dir/xvectors_sitw_dev_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#    "cat '$sitw_dev_trials_core' | cut -d\  --fields=1,2 |" $nnet_dir/scores/sitw_dev_core_scores || exit 1;
#
#  echo "SITW Dev Core:"
#  eer=$(paste $sitw_dev_trials_core $nnet_dir/scores/sitw_dev_core_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
#  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $nnet_dir/scores/sitw_dev_core_scores $sitw_dev_trials_core 2> /dev/null`
#  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnet_dir/scores/sitw_dev_core_scores $sitw_dev_trials_core 2> /dev/null`
#  echo "EER: $eer%"
#  echo "minDCF(p-target=0.01): $mindcf1"
#  echo "minDCF(p-target=0.001): $mindcf2"
#fi