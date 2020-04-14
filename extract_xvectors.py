import argparse
import configparser
import os
import pickle
import sys
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import torch

from kaldiio import WriteHelper
import kaldi_io

from data_io_speaker import SpeakerDataset, SpeakerTestDataset
from models_speaker import ETDNN, FTDNN, XTDNN

def parse_args():
    parser = argparse.ArgumentParser(description='extract embeddings')
    parser.add_argument('--cfg', type=str, default='./configs/example_speaker.cfg')
    parser.add_argument('--best', action='store_true', default=False, help='Use best model')
    parser.add_argument('--checkpoint', type=int, default=-1, # which model to use, overidden by 'best'
                            help='Use model checkpoint, default -1 uses final model')
    args = parser.parse_args()
    assert os.path.isfile(args.cfg)
    return args


def parse_config(args):
    config = configparser.ConfigParser()
    config.read(args.cfg)

    args.train_data = config['Datasets']['train']
    args.train_data = args.train_data.replace("no_sil", "200k")
    print('TRAIN dataset: {}'.format(args.train_data))
    args.test_data_vc1 = config['Datasets'].get('test_vc1')
    print('VC1 dataset: {}'.format(args.test_data_vc1))
    args.test_data_sitw = config['Datasets'].get('test_sitw')
    print('SITW dataset eval: {}'.format(args.test_data_sitw))

    args.no_cuda = config['Hyperparams'].getboolean('no_cuda', fallback=False)

    args.model_type = config['Model'].get('model_type', fallback='XTDNN')
    assert args.model_type in ['XTDNN', 'ETDNN', 'FTDNN']

    args.num_iterations = config['Hyperparams'].getint('num_iterations', fallback=50000)

    args.model_dir = config['Outputs']['model_dir']
    return args

def extract_train(generator, ds_train, device, path):
    generator.eval()
    num_examples = len(ds_train)
    ark_scp_xvector ='ark:| copy-vector ark:- ark,scp:{0}/xvector.ark,{0}/xvector.scp'.format(path)

    #with WriteHelper('ark,scp:{0}/xvector.ark,{0}/xvector.scp'.format(path)) as writer:
    with kaldi_io.open_or_fd(ark_scp_xvector,'w') as f:
        with torch.no_grad():
            for i in tqdm(range(num_examples)):
                feats, utt = ds_train.__getitem__(i)
                feats = feats.unsqueeze(0).to(device)
                embeds = generator(feats).cpu().numpy()
                #writer(utt, embeds[0])
                kaldi_io.write_vec_flt(f, embeds[0], key=utt)

def extract_test(generator, ds_test, device, path):
    generator.eval()
    num_examples = len(ds_test.veri_utts)
    ark_scp_xvector ='ark:| copy-vector ark:- ark,scp:{0}/xvector.ark,{0}/xvector.scp'.format(path)

    spks = []
    spk_mean = {}
    spk_count = {}
    #with WriteHelper('ark,scp:{0}/xvector.ark,{0}/xvector.scp'.format(path)) as writer:
    with kaldi_io.open_or_fd(ark_scp_xvector,'w') as f:
        with torch.no_grad():
            for i in tqdm(range(num_examples)):
                feats, utt = ds_test.__getitem__(i)
                feats = feats.unsqueeze(0).to(device)
                embeds = generator(feats).cpu().numpy()
                #writer(utt, embeds[0])
                kaldi_io.write_vec_flt(f, embeds[0], key=utt)

                spk = ds_test.utt2spk_dict[utt]
                if spk not in spk_mean:
                    spk_mean[spk] = embeds[0]
                    spk_count[spk] = 1
                    spks.append(spk)
                else:
                    spk_mean[spk] = spk_mean[spk] + embeds[0]
                    spk_count[spk] += 1

    ark_scp_spk_xvector ='ark:| copy-vector ark:- ark,scp:{0}/spk_xvector.ark,{0}/spk_xvector.scp'.format(path)
    with kaldi_io.open_or_fd(ark_scp_spk_xvector,'w') as f:
        for spk in spks:
            mean = spk_mean[spk] / spk_count[spk]
            kaldi_io.write_vec_flt(f, mean, key=spk)
    
    ark_num_utts ='{0}/num_utts.ark'.format(path)
    with open(ark_num_utts,'w') as f:
        for spk in spks:
            f.write(spk + " " + str(spk_count[spk]) + "\n")

if __name__ == "__main__":
    args = parse_args()
    args = parse_config(args)


    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_cuda = False # !!!!
    print('='*30)
    print('USE_CUDA SET TO: {}'.format(use_cuda))
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()))
    print('='*30)
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.checkpoint == -1:
        g_path = os.path.join(args.model_dir, "final_g_{}.pt".format(args.num_iterations))
        g_path_sitw = g_path
        g_path_vc1 = g_path
    else:
        g_path = os.path.join(args.model_dir, "g_{}.pt".format(args.checkpoint))
        g_path_sitw = g_path
        g_path_vc1 = g_path
    
    if args.model_type == 'XTDNN':
        generator = XTDNN()
    if args.model_type == 'ETDNN':
        generator = ETDNN()
    if args.model_type == 'FTDNN':
        generator = FTDNN()

    if args.best:
        args.results_pkl = os.path.join(args.model_dir, 'results.p')
        rpkl = pickle.load(open(args.results_pkl, "rb"))
        
        if args.test_data_vc1:
            v1eers = [(rpkl[key]['vc1_eer'], key) for key in rpkl]
            best_vc1_cp = min(v1eers)[1]
            g_path_vc1 = os.path.join(args.model_dir, "g_{}.pt".format(best_vc1_cp))
            print('Best VC1 Model: {}'.format(g_path_vc1))


        if args.test_data_sitw:
            sitweers = [(rpkl[key]['sitw_eer'], key) for key in rpkl]
            best_sitw_cp = min(sitweers)[1]
            g_path_sitw = os.path.join(args.model_dir, "g_{}.pt".format(best_sitw_cp))
            print('Best SITW Model: {}'.format(g_path_sitw))


    # load 200k utt from plda training
    train_path = '{}/xvectors_train_combined_200k'.format(args.model_dir)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    ds_train = SpeakerDataset(args.train_data)
    generator.load_state_dict(torch.load(g_path))
    generator = generator.to(device)
    extract_train(generator, ds_train, device, train_path)

    if args.test_data_vc1:
        vc1_path = '{}/xvectors_voxceleb1'.format(args.model_dir)
        if not os.path.exists(vc1_path):
            os.makedirs(vc1_path)
        ds_test_vc1 = SpeakerTestDataset(args.test_data_vc1)
        generator.load_state_dict(torch.load(g_path_vc1))
        generator = generator.to(device)
        extract_test(generator, ds_test_vc1, device, vc1_path)
    
    #if args.test_data_sitw:
    #    ds_test_sitw = SpeakerTestDataset(args.test_data_sitw)
    #    generator.load_state_dict(torch.load(g_path_sitw))
    #    generator = generator.to(device)
    #    sitw_eer, sitw_mdcf1, sitw_mdcf2 = test_nosil(generator, ds_test_sitw, device, mindcf=True)
    #    print("="*60)
    #    print('SITW(dev):: \t EER: {}, minDCF(p=0.01): {}, minDCF(p=0.001): {}'.format(round_sig(sitw_eer, 3), round_sig(sitw_mdcf1, 3), round_sig(sitw_mdcf2, 3)))
    #    print("="*60)