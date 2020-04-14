import argparse
import configparser
import glob
import json
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict
from pprint import pprint

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import uvloop
from data_io_speaker import SpeakerDataset, SpeakerTestDataset
from loss_functions import (AdaCos, AMSMLoss, DisturbLabelLoss, L2SoftMax,
                            LabelSmoothingLoss, XVecHead, SoftMax, ArcFace, SphereFace)
from models_speaker import ETDNN, FTDNN, XTDNN
from test_model_speaker import test, test_nosil
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (SpeakerRecognitionMetrics, schedule_lr)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def parse_args():
    parser = argparse.ArgumentParser(description='Train SV model')
    parser.add_argument('--cfg', type=str, default='./configs/example_speaker.cfg')
    parser.add_argument('--resume-checkpoint', type=int, default=0)
    args = parser.parse_args()
    assert os.path.isfile(args.cfg)
    args._start_time = time.ctime()
    return args


def parse_config(args):
    config = configparser.ConfigParser()
    config.read(args.cfg)

    args.train_data = config['Datasets']['train']
    assert args.train_data
    args.test_data_vc1 = config['Datasets'].get('test_vc1')
    print('VC1 dataset: {}'.format(args.test_data_vc1))
    args.test_data_sitw = config['Datasets'].get('test_sitw')
    print('SITW dataset: {}'.format(args.test_data_sitw))

    args.model_type = config['Model'].get('model_type', fallback='XTDNN')
    assert args.model_type in ['XTDNN', 'ETDNN', 'FTDNN']

    args.loss_type = config['Optim'].get('loss_type', fallback='adacos')
    assert args.loss_type in ['l2softmax', 'adm', 'adacos', 'xvec', 'arcface', 'sphereface', 'softmax']
    args.label_smooth_type = config['Optim'].get('label_smooth_type', fallback='None')
    assert args.label_smooth_type in ['None', 'disturb', 'uniform']
    args.label_smooth_prob = config['Optim'].getfloat('label_smooth_prob', fallback=0.1)

    args.lr = config['Hyperparams'].getfloat('lr', fallback=0.2)
    args.batch_size = config['Hyperparams'].getint('batch_size', fallback=400)
    args.max_seq_len = config['Hyperparams'].getint('max_seq_len', fallback=400)
    args.no_cuda = config['Hyperparams'].getboolean('no_cuda', fallback=False)
    args.seed = config['Hyperparams'].getint('seed', fallback=123)
    args.num_iterations = config['Hyperparams'].getint('num_iterations', fallback=50000)
    args.momentum = config['Hyperparams'].getfloat('momentum', fallback=0.9)
    args.scheduler_steps = np.array(json.loads(config.get('Hyperparams', 'scheduler_steps'))).astype(int)
    args.scheduler_lambda = config['Hyperparams'].getfloat('scheduler_lambda', fallback=0.5)
    args.multi_gpu = config['Hyperparams'].getboolean('multi_gpu', fallback=False)
    args.classifier_lr_mult = config['Hyperparams'].getfloat('classifier_lr_mult', fallback=1.)
    args.log_interval = config['Hyperparams'].getfloat('log_interval', fallback=100)

    args.model_dir = config['Outputs']['model_dir']
    args.log_file = os.path.join(args.model_dir, 'train.log')
    args.checkpoint_interval = config['Outputs'].getint('checkpoint_interval')
    args.results_pkl = os.path.join(args.model_dir, 'results.p')

    return args


def train(ds_train):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('\n' + '=' * 30)
    print('USE_CUDA SET TO: {}'.format(use_cuda))
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()))
    print('=' * 30 + '\n')
    device = torch.device("cuda" if use_cuda else "cpu")

    writer = SummaryWriter(comment=os.path.basename(args.cfg))
    num_classes = ds_train.num_classes

    if args.model_type == 'XTDNN':
        generator = XTDNN()
    if args.model_type == 'ETDNN':
        generator = ETDNN()
    if args.model_type == 'FTDNN':
        generator = FTDNN()

    if args.loss_type == 'adm':
        classifier = AMSMLoss(512, num_classes)
    if args.loss_type == 'adacos':
        classifier = AdaCos(512, num_classes)
    if args.loss_type == 'l2softmax':
        classifier = L2SoftMax(512, num_classes)
    if args.loss_type == 'softmax':
        classifier = SoftMax(512, num_classes)
    if args.loss_type == 'xvec':
        classifier = XVecHead(512, num_classes)
    if args.loss_type == 'arcface':
        classifier = ArcFace(512, num_classes)
    if args.loss_type == 'sphereface':
        classifier = SphereFace(512, num_classes)

    generator.train()
    classifier.train()

    generator = generator.to(device)
    classifier = classifier.to(device)

    if args.resume_checkpoint != 0:
        model_str = os.path.join(args.model_dir, '/checkpoints/{}_{}.pt')
        for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
            model.load_state_dict(torch.load(model_str.format(modelstr, args.resume_checkpoint)))

    optimizer = torch.optim.SGD([{'params': generator.parameters(), 'lr': args.lr},
                                 {'params': classifier.parameters(), 'lr': args.lr * args.classifier_lr_mult}],
                                momentum=args.momentum)

    if args.label_smooth_type == 'None':
        criterion = nn.CrossEntropyLoss()
    if args.label_smooth_type == 'disturb':
        criterion = DisturbLabelLoss(device, disturb_prob=args.label_smooth_prob)
    if args.label_smooth_type == 'uniform':
        criterion = LabelSmoothingLoss(smoothing=args.label_smooth_prob)

    iterations = 0

    total_loss = 0
    running_loss = [np.nan for _ in range(500)]

    best_vc1_eer = (-1, 1.0)
    best_sitw_eer = (-1, 1.0)

    if os.path.isfile(args.results_pkl):
        rpkl = pickle.load(open(args.results_pkl, "rb"))
        if args.test_data_vc1:
            v1eers = [(rpkl[key]['vc1_eer'], i) for i, key in enumerate(rpkl)]
            bestvc1 = min(v1eers)
            best_vc1_eer = (bestvc1[1], bestvc1[0])
        if args.test_data_sitw:
            sitweers = [(rpkl[key]['sitw_eer'], i) for i, key in enumerate(rpkl)]
            bestsitw = min(sitweers)
            best_sitw_eer = (bestsitw[1], bestsitw[0])
    else:
        rpkl = OrderedDict({})

    if args.multi_gpu:
        dpp_generator = nn.DataParallel(generator).to(device)

    data_generator = ds_train.get_batches(batch_size=args.batch_size, max_seq_len=args.max_seq_len)

    if args.model_type == 'FTDNN':
        drop_indexes = np.linspace(0, 1, args.num_iterations)
        drop_sch = ([0, 0.5, 1], [0, 0.5, 0])
        drop_schedule = np.interp(drop_indexes, drop_sch[0], drop_sch[1])

    for iterations in range(1, args.num_iterations + 1):
        if iterations > args.num_iterations:
            break
        if iterations in args.scheduler_steps:
            schedule_lr(optimizer, factor=args.scheduler_lambda)
        if iterations <= args.resume_checkpoint:
            print('Skipping iteration {}'.format(iterations))
            print('Skipping iteration {}'.format(iterations), file=open(args.log_file, "a"))
            continue

        if args.model_type == 'FTDNN':
            generator.set_dropout_alpha(drop_schedule[iterations - 1])

        feats, iden = next(data_generator)
        feats = feats.to(device)

        iden = torch.LongTensor(iden).to(device)

        if args.multi_gpu:
            embeds = dpp_generator(feats)
        else:
            embeds = generator(feats)

        if args.loss_type == 'softmax':
            preds = classifier(embeds)
        else:
            preds = classifier(embeds, iden)

        loss = criterion(preds, iden)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.model_type == 'FTDNN':
            generator.step_ftdnn_layers()

        running_loss.pop(0)
        running_loss.append(loss.item())
        rmean_loss = np.nanmean(np.array(running_loss))

        if iterations % args.log_interval == 0:
            msg = "{}: {}: [{}/{}] \t C-Loss:{:.4f}, AvgLoss:{:.4f}, lr: {}, bs: {}".format(args.model_dir,
                                                                                            time.ctime(),
                                                                                            iterations,
                                                                                            args.num_iterations,
                                                                                            loss.item(),
                                                                                            rmean_loss,
                                                                                            get_lr(optimizer),
                                                                                            len(feats))
            print(msg)
            print(msg, file=open(args.log_file, "a"))

        writer.add_scalar('class loss', loss.item(), iterations)
        writer.add_scalar('Avg loss', rmean_loss, iterations)

        if iterations % args.checkpoint_interval == 0:
            for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
                model.eval().cpu()
                cp_filename = "{}_{}.pt".format(modelstr, iterations)
                cp_model_path = os.path.join(args.model_dir, cp_filename)
                torch.save(model.state_dict(), cp_model_path)
                model.to(device).train()

            rpkl[iterations] = {}

            if args.test_data_vc1:
                vc1_eer = test(generator, ds_test_vc1, device)
                print('EER on VoxCeleb1: {}'.format(vc1_eer))
                print('EER on Voxceleb1: {}'.format(vc1_eer), file=open(args.log_file, "a"))
                writer.add_scalar('vc1_eer', vc1_eer, iterations)
                if vc1_eer < best_vc1_eer[1]:
                    best_vc1_eer = (iterations, vc1_eer)
                print('Best VC1 EER: {}'.format(best_vc1_eer))
                print('Best VC1 EER: {}'.format(best_vc1_eer), file=open(args.log_file, "a"))
                rpkl[iterations]['vc1_eer'] = vc1_eer

            if args.test_data_sitw:
                sitw_eer = test_nosil(generator, ds_test_sitw, device)
                print('EER on SITW: {}'.format(sitw_eer))
                print('EER on SITW: {}'.format(sitw_eer), file=open(args.log_file, "a"))
                writer.add_scalar('sitw_eer', sitw_eer, iterations)
                if sitw_eer < best_sitw_eer[1]:
                    best_sitw_eer = (iterations, sitw_eer)
                print('Best SITW EER: {}'.format(best_sitw_eer))
                print('Best SITW EER: {}'.format(best_sitw_eer), file=open(args.log_file, "a"))
                rpkl[iterations]['sitw_eer'] = sitw_eer

            pickle.dump(rpkl, open(args.results_pkl, "wb"))

    # ---- Final model saving -----
    for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
        model.eval().cpu()
        cp_filename = "final_{}_{}.pt".format(modelstr, iterations)
        cp_model_path = os.path.join(args.model_dir, cp_filename)
        torch.save(model.state_dict(), cp_model_path)

if __name__ == "__main__":
    args = parse_args()
    args = parse_config(args)
    os.makedirs(args.model_dir, exist_ok=True)
    if args.resume_checkpoint == 0:
        shutil.copy(args.cfg, os.path.join(args.model_dir, 'experiment_settings.cfg'))
    else:
        shutil.copy(args.cfg, os.path.join(args.model_dir, 'experiment_settings_resume.cfg'))
    if os.path.exists(args.log_file):
        os.remove(args.log_file)
    pprint(vars(args))
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    uvloop.install()
    ds_train = SpeakerDataset(args.train_data)
    if args.test_data_vc1:
        ds_test_vc1 = SpeakerTestDataset(args.test_data_vc1)
    if args.test_data_sitw:
        ds_test_sitw = SpeakerTestDataset(args.test_data_sitw)

    train(ds_train)
