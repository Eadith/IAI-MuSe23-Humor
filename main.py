import argparse
import os
import random
import sys
from datetime import datetime

import numpy
import torch
from dateutil import tz

import config
from config import TASKS, PERSONALISATION, HUMOR, MIMIC, AROUSAL, VALENCE, PERSONALISATION_DIMS
from data_parser import load_data
from dataset import MuSeDataset, custom_collate_fn
from eval import evaluate, calc_ccc, calc_auc, mean_pearsons
from loss import CCCLoss, BCELossWrapper, MSELossWrapper, BCEWithLogitsLossWrapper, Focal_Loss
from model import JTMA, JTMA_Ensemble
from train import train_model
from utils import Logger, seed_worker, log_results


def parse_args():

    parser = argparse.ArgumentParser(description='MuSe 2023.')

    parser.add_argument('--task', type=str, choices=TASKS, default="humor",    # required=True
                        help=f'Specify the task from {TASKS}.')
    # parser.add_argument('--feature', default="w2v-msp",    #  required=True
    #                     help='Specify the features used (only one).')
    # parser.add_argument('--feature', nargs='+', #default=['affect_vit_face'], 
    #                      default=['affect_vit_face','affect_face','egemaps','ds','w2v-msp',
    #                      'faus','vit','facenet','bert-multilingual'], 
    #                     help='Specify the features used (multi-modality).')
    parser.add_argument('--feature', nargs='+', # default=['bert-multilingual-sentiment2'], 
                        #   default=['affect_vit_face'], 
                        #   default = ["affect_vit_face_aug"],
                        #   default = ["affect_vit_face", "w2v-msp"],
                        #    default = ["affect_vit_face", "bert-multilingual"],
                        #    default = ["affect_vit_face", "affect_vit_face_aug", "w2v-msp"],
                        #   default = ["affect_vit_face", "affect_vit_face_aug", "bert-multilingual"],
                        #    default = ["affect_vit_face", "w2v-msp", "bert-multilingual"],
                        #   default = ["affect_vit_face_aug", "w2v-msp", "bert-multilingual"],
                        #   default = ["affect_vit_face_aug", "w2v-msp", "bert-multilingual", "affect_vit_face"],
                           default = [ "w2v-msp", "bert-multilingual", "affect_vit_face"],
                        help='Specify the features used (multi-modality).')

                        

    parser.add_argument('--emo_dim', default=AROUSAL, choices=PERSONALISATION_DIMS,
                        help=f'Specify the emotion dimension, only relevant for personalisation (default: {AROUSAL}).')
    parser.add_argument('--normalize', action='store_true',
                        help='Specify whether to normalize features (default: False).')
    parser.add_argument('--win_len', type=int, default=200,
                        help='Specify the window length for segmentation (default: 200 frames).')
    parser.add_argument('--hop_len', type=int, default=100,
                        help='Specify the hop length to for segmentation (default: 100 frames).')
    
    
    parser.add_argument('--model_dim', type=int, default=64,
                        help='Specify the number of hidden states in the RNN (default: 64).')
    parser.add_argument('--rnn_n_layers', type=int, default=1,
                        help='Specify the number of layers for the RNN (default: 1).')
    parser.add_argument('--rnn_bi', action='store_true',
                        help='Specify whether the RNN is bidirectional or not (default: False).')
    parser.add_argument('--d_fc_out', type=int, default=64,
                        help='Specify the number of hidden neurons in the output layer (default: 64).')
    parser.add_argument('--rnn_dropout', type=float, default=0.2)
    parser.add_argument('--linear_dropout', type=float, default=0.5)


    parser.add_argument('--epochs', type=int, default=100,
                        help='Specify the number of epochs (default: 100).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Specify the batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Specify initial learning rate (default: 0.0001).')
    parser.add_argument('--seed', type=int, default=3047,
                        help='Specify the initial random seed (default: 101).')
    parser.add_argument('--n_seeds', type=int, default=10,
                        help='Specify number of random seeds to try (default: 5).')
    parser.add_argument('--result_csv', default=None, help='Append the results to this csv (or create it, if it '
                                                           'does not exist yet). Incompatible with --predict')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--reduce_lr_patience', type=int, default=5, help='Patience for reduction of learning rate')
    parser.add_argument('--use_gpu', action='store_false',       # action='store_true',
                        help='Specify whether to use gpu for training (default: True).')
    parser.add_argument('--cache', action='store_false',    # action='store_true',
                        help='Specify whether to cache data as pickle file (default: True).')
    parser.add_argument('--predict', action='store_false',    # action='store_true',
                        help='Specify when no test labels are available; test predictions will be saved '
                             '(default: False). Incompatible with result_csv')
    parser.add_argument('--regularization', type=float, required=False, default=0.0,
                        help='L2-Penalty')
    
    
    # evaluation only arguments
    parser.add_argument('--eval_model', type=str,   default="TEMMA_2023-06-28-14-10_[affect_vit_face-w2v-msp-bert-multilingual]_[0.0001_256]",
                        help='Specify model which is to be evaluated; no training with this option (default: False).')
    parser.add_argument('--eval_seed', type=str,   default="103",
                        help='Specify seed to be evaluated; only considered when --eval_model is given.')

    
    # TEMMA parameters
    parser.add_argument('--block_num', type=int, default=4)   # 4
    parser.add_argument('--dropout', type=float, default=0.2)    # default=0.2
    parser.add_argument('--dropout_mmatten', type=float, default=0.5)    # default=0.5
    parser.add_argument('--dropout_mtatten', type=float, default=0.2)    # default=0.2
    parser.add_argument('--dropout_ff', type=float, default=0.2)    # default=0.2
    parser.add_argument('--dropout_subconnect', type=float, default=0.2)    # default=0.2
    parser.add_argument('--dropout_position', type=float, default=0.2)    # default=0.2
    parser.add_argument('--dropout_embed', type=float, default=0.2)    # default=0.2
    parser.add_argument('--dropout_fc', type=float, default=0.2)    # default=0.2
    parser.add_argument('--h', type=int, default=4)
    parser.add_argument('--h_mma', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--embed', type=str, default='temporal')
    parser.add_argument('--levels', type=int, default=5)
    parser.add_argument('--ksize', type=int, default=3)
    parser.add_argument('--ntarget', type=int, default=1)


    args = parser.parse_args()
    if not (args.result_csv is None) and args.predict:
        print("--result_csv is not compatible with --predict")
        sys.exit(-1)
    if args.eval_model:
        assert args.eval_seed
    return args


def get_loss_fn(task, alpha=0.25, gamma=2):
    if task == PERSONALISATION:
        return CCCLoss(), 'CCC'
    elif task == HUMOR:
        return Focal_Loss(alpha=alpha, gamma=gamma), 'Binary Crossentropy'    # 此处修改解决数据不平衡问题
    elif task == MIMIC:
        return MSELossWrapper(reduction='mean'), 'MSE'




def get_eval_fn(task):
    if task == PERSONALISATION:
        return calc_ccc, 'CCC'
    elif task == MIMIC:
        return mean_pearsons, 'Mean Pearsons'
    elif task == HUMOR:
        return calc_auc, 'AUC-Score'


def main(args):
    # ensure reproducibility
    numpy.random.seed(10)
    random.seed(10)
    torch.manual_seed(args.seed)

    print(args.feature)

    # emo_dim only relevant for stress/personalisation
    args.emo_dim = args.emo_dim if args.task==PERSONALISATION else ''
    print('Loading data ...')
    # data = load_data(args.task, args.paths, args.feature, args.emo_dim, args.normalize,
    #                  args.win_len, args.hop_len, save=args.cache)
    # datasets = {partition:MuSeDataset(data, partition) for partition in data.keys()}

    data = []
    for name in args.feature:
        if name == "egemaps" or name == "vit":    # 此处需要修改，如果后续特征需要normalize，应该在这里添加！！
            args.normalize = True
        data.append(load_data(args.task, args.paths, name, args.emo_dim, args.normalize, args.win_len, args.hop_len, save=args.cache))

    # data --> dataset --> data_loader
    data_loader = {}
    for partition in data[0].keys():     # train, eval and test.
        dataset = MuSeDataset(data, partition)    # modify MuSeDataset to adapt multi-modality input.
        batch_size = args.batch_size if partition == 'train' else (1 if args.task==PERSONALISATION else 2*args.batch_size)
        shuffle = True if partition == 'train' else False   # shuffle only for train partition
        data_loader[partition] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4,
                                                             worker_init_fn=seed_worker, collate_fn=custom_collate_fn)
    

    # args.d_in = datasets['train'].get_feature_dim()
    args.d_in = data_loader['train'].dataset.get_feature_dim()


    args.n_targets = config.NUM_TARGETS[args.task]
    args.n_to_1 = args.task in config.N_TO_1_TASKS

    loss_fn, loss_str = get_loss_fn(args.task, alpha=0.8125, gamma=0.5)           # 并没有解决数据不平衡问题(已经解决)  -->  focus loss
    eval_fn, eval_str = get_eval_fn(args.task)

    if args.eval_model is None:  # Train and validate for each seed
        seeds = range(args.seed, args.seed + args.n_seeds)
        val_losses, val_scores, best_model_files, test_scores = [], [], [], []

        for seed in seeds:
            # move data initialisation below here...
            torch.manual_seed(seed)

            # data_loader = {}
            # for partition,dataset in datasets.items():  # one DataLoader for each partition
            #     batch_size = args.batch_size if partition == 'train' else (
            #         1 if args.task == PERSONALISATION else 2 * args.batch_size)
            #     shuffle = True if partition == 'train' else False  # shuffle only for train partition
            #     data_loader[partition] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
            #                                                          num_workers=4,
            #                                                          worker_init_fn=seed_worker,
            #                                                          collate_fn=custom_collate_fn)

            # model = Model(args)
            
            model = JTMA(args, args.d_in, data_loader['train'].dataset.get_feature_len_in_one_step())

            print('=' * 50)
            print(f'Training model... [seed {seed}] for at most {args.epochs} epochs')

            val_loss, val_score, best_model_file = train_model(args.task, model, data_loader, args.epochs,
                                                               args.lr, args.paths['model'], seed, use_gpu=args.use_gpu,
                                                               loss_fn=loss_fn, eval_fn=eval_fn,
                                                               eval_metric_str=eval_str,
                                                               regularization=args.regularization,
                                                               early_stopping_patience=args.early_stopping_patience,
                                                               reduce_lr_patience=args.reduce_lr_patience)
            # restore best model encountered during training
            model = torch.load(best_model_file)

            if not args.predict:  # run evaluation only if test labels are available
                test_loss, test_score = evaluate(args.task, model, data_loader['test'], loss_fn=loss_fn,
                                                 eval_fn=eval_fn, use_gpu=args.use_gpu)
                test_scores.append(test_score)
                print(f'[Test {eval_str}]:  {test_score:7.4f}')
            val_losses.append(val_loss)
            val_scores.append(val_score)
            best_model_files.append(best_model_file)

    
        best_idx = val_scores.index(max(val_scores))  # find best performing seed

        print('=' * 50)
        print(f'Best {eval_str} on [Val] for seed {seeds[best_idx]}: '
              f'[Val {eval_str}]: {val_scores[best_idx]:7.4f}'
              f"{f' | [Test {eval_str}]: {test_scores[best_idx]:7.4f}' if not args.predict else ''}")
        print('=' * 50)


        # TEMMA-Ensemble
        # ensemble_model = TEMMA_Ensemble(args, best_model_files, len(best_model_files))

        # print('=' * 50)
        # print(f'Training model...  for at most {args.epochs} epochs')

        # val_loss, val_score, best_model_file = train_model(args.task, model, data_loader, args.epochs,
        #                                                     args.lr, args.paths['model'], seed, use_gpu=args.use_gpu,
        #                                                     loss_fn=loss_fn, eval_fn=eval_fn,
        #                                                     eval_metric_str=eval_str,
        #                                                     regularization=args.regularization,
        #                                                     early_stopping_patience=args.early_stopping_patience,
        #                                                     reduce_lr_patience=args.reduce_lr_patience)


        model_file = best_model_files[best_idx]  # best model of all of the seeds
        if not args.result_csv is None:
            log_results(args.result_csv, params=args, seeds = list(seeds), metric_name=eval_str,
                        model_files=best_model_files, test_results=test_scores, val_results=val_scores,
                        best_idx=best_idx)

    else:  # Evaluate existing model (No training)
        model_file = os.path.join(args.paths['model'], f'model_{args.eval_seed}.pth')
        model = torch.load(model_file, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        # data_loader = {}
        # for partition, dataset in datasets.items():  # one DataLoader for each partition
        #     batch_size = args.batch_size if partition == 'train' else (
        #         1 if args.task == PERSONALISATION else 2 * args.batch_size)
        #     shuffle = True if partition == 'train' else False  # shuffle only for train partition
        #     data_loader[partition] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        #                                                          num_workers=4,
        #                                                          worker_init_fn=seed_worker,
        #                                                          collate_fn=custom_collate_fn)
        _, valid_score = evaluate(args.task, model, data_loader['devel'], loss_fn=loss_fn, eval_fn=eval_fn,
                                  use_gpu=args.use_gpu)
        print(f'Evaluating {model_file}:')
        print(f'[Val {eval_str}]: {valid_score:7.4f}')
        if not args.predict:
            _, test_score = evaluate(args.task, model, data_loader['test'], loss_fn=loss_fn, eval_fn=eval_fn,
                                     use_gpu=args.use_gpu)
            print(f'[Test {eval_str}]: {test_score:7.4f}')

    if args.predict:  # Make predictions for the test partition; this option is set if there are no test labels
        print('Predicting devel and test samples...')
        best_model = torch.load(model_file, map_location=config.device)
        evaluate(args.task, best_model, data_loader['devel'], loss_fn=loss_fn, eval_fn=eval_fn,
                 use_gpu=args.use_gpu, predict=True, prediction_path=args.paths['predict'],
                 filename='predictions_devel.csv')
        evaluate(args.task, best_model, data_loader['test'], loss_fn=loss_fn, eval_fn=eval_fn,
                 use_gpu=args.use_gpu, predict=True, prediction_path=args.paths['predict'], filename='predictions_test.csv')
        print(f'Find predictions in {os.path.join(args.paths["predict"])}')

    print('Done.')


if __name__ == '__main__':
    args = parse_args()

    args.log_file_name = '{}_{}_[{}]_[{}]_[{}_{}_{}_{}]_[{}_{}]'.format('RNN',
        datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"), args.feature, args.emo_dim,
        args.model_dim, args.rnn_n_layers, args.rnn_bi, args.d_fc_out, args.lr, args.batch_size) if args.task == PERSONALISATION else \
        '{}_{}_[{}]_[{}_{}]'.format('WIN', datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"), "-".join(args.feature), args.lr,args.batch_size)   # 模型参数在这里添加以下，用于区分

    # adjust your paths in config.py
    args.paths = {'log': os.path.join(config.LOG_FOLDER, args.task) if not args.predict else os.path.join(config.LOG_FOLDER, args.task, 'prediction'),
                  'data': os.path.join(config.DATA_FOLDER, args.task),
                  'model': os.path.join(config.MODEL_FOLDER, args.task, args.log_file_name if not args.eval_model else args.eval_model)}
    
    if args.predict:
        if args.eval_model:
            args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, args.task, args.eval_model, args.eval_seed)
        else:
            args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, args.task, args.log_file_name)

    for folder in args.paths.values():
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    args.paths.update({'features': config.PATH_TO_FEATURES[args.task],
                       'labels': config.PATH_TO_LABELS[args.task],
                       'partition': config.PARTITION_FILES[args.task]})

    sys.stdout = Logger(os.path.join(args.paths['log'], args.log_file_name + '.txt'))
    print(' '.join(sys.argv))

    # 搜索 focus loss 超参
    # gamma = numpy.linspace(0.5, 5, num=5, endpoint=True, retstep=False, dtype=None)
    # alpha = numpy.linspace(0.25, 1, num=4, endpoint=False)
    # for g in gamma:
    #     for a in alpha:
    #         args.gamma = g
    #         args.alpha = a
    #         print("gamma={}, alpha={}".format(g, a))
    #         main(args)
    # for dropout_fc in [0.2, 0.4, 0.5, 0.6, 0.8]:
    #     args.dropout_fc = dropout_fc 
    #     print(args.dropout_fc)
    #     print(len(args.feature))
    #     print(args.feature)
    #     main(args)
    main(args)

