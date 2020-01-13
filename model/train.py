"""
train.py

train: using model_layers.py and model.py BiDAF model will train given SQuAD dataset.
"""



import numpy as np
import argparse
import util
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import torch.optimizer.lr_scheduler as lr_sched
import torch.utils.data as data

from collections import OrderedDict
from json import dumps
from model import BiDAF
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load

# change these names
from util import collate_fn, SQuAD

def get_train_args():
    """Get arguments needed in train.py."""
    parser = argparse.ArgumentParser('Train a model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--eval_steps',
                        type=int,
                        default=50000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.5,
                        help='Learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('NLL', 'EM', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')

    args = parser.parse_args()

    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    return args


# string model_dir: model directory to create save Directory
# string model_name: name to identify model uniquely
# bool is_training: training or evaluate
# Returns: str save_dir: path where new model will be saved
def get_save_dir(save_dir, model_name, is_training):
    for uniq_name in range(100):
        subdir = 'train'if training else 'test'
        save_dir = os.path.join(model_dir, subdirb, f'{model_name}_{uniq_name:02d}')
        if not os.path.exists(save_dir):
            return save_dir

def main(args):

    # setup of saving Directory
    # setup of tensorboard
    # checking for GPUs
    args.save_dir = get_save_dir(args.save_dir, args.name, is_training=True)
    log = util.get_logger(args.save_dir, args.name)
    tboard = SummaryWriter(args.save_dir)
    device, args.gpu_list = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_list))

    # setting random seed
    log.info(f'Random Seed: {args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(agrs.seed)

    # get embeddings
    log.info('Loading Embeddings')
    w_vectors = util.tensor_from_json(args.word_emb_file)

    log.info('Model Initialising')
    model = BiDAF(word_vectors, args.hidden_size, args.drop_prob)

    # parallelise data if multiple devices detected
    model = nn.DataParallel(model, args.gpu_list)

    # loading weights from checkpoint if available
    if args.load_path:
        log.info(f'Loading checkpoint {args.load_path}')
        model, step = util.load_model(mdoel, args.load_path, args.gpu_list)
    else:
        log.info(f'Found no steps starting from step 0')
        step = 0

    model = model.to(device)
    mdoel.train()
    ema = util.ema(model, args.ema_decay)

    # set up checkpoint saving
    saver = util.CheckpointSaver(args.save_dir, args.max_checkpoints, args.metric_name, args.maximize_metric, log)

    # optimizer
    ada_optimzer = optimizer.Adadelta(model.parameters(), args.lr, weight_decay = args.l2_wd)

    # lambda can change defualt is constant
    lr_schedule = lr_sched.LambdaLR(ada_optimzer, lambda s: 1.)

    log.info('Loading Dataset')

    # training dataset
    train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    train_loader = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers,m collate_fn = collate_fn)

    # eval dataset
    eval_dataset = SQuAD(args.dev_record_file,args.use_squad_v2)
    eval_loader = data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, collate_fn = collate_fn)

    # Train
    log.info('Traing is beginging')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)

    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Epoch {epoch}')
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            # cc_idx is context2char which is not used in this model
            # cw_idx: context2word as input to model
            # qw_idx: query2word as input to model
            for cw_idx, cc_idx, qw_idx, qc_idx, y1, y2, ids in train_loader:

                # initialising Forward
                cw_idx = cw_idx.to(device)
                qw_idx = qw_idx.to(device)
                batch_size = cw_idx.size(0)
                # clearing all gradients
                ada_optimzer.zero_grad()

                # forward
                log_p1, log_p2 = model(cw_idx, qw_idx)
                y1 = y1.to(device)
                y2 = y2.to(device)

                # calulating loss using
                # Poisson Negative log likelihood
                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                loss_val = loss.item()

                # backward
                loss.backward()

                # calculates total norm of all parameters into a vector
                nn.utils.clip_grad_norm(model.parameters(),args.max_grad_norm)
                ada_optimzer.step()
                lr_schedule.step(step // batch_size)

                ema(model, step // batch_size)

                # Log info
                step += batch_size
                # tqmd for visual progress bar
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch = epoch, NLL = loss_val)

                # updating tensorboard
                tboard.add_scalar('train/Neg Log Likelihood', loss_val, step)
                tboard.add_scalar('train/Learning Rate',ada_optimzer.param_groups[0]['lr'] , step)


                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps


                    # evaluate model
                    ema.assign(model)
                    results, pred_dict = evaluate(model, model, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)
                    saver.save(step, model, results[args.metric_name], device)

                    # returning all parameters back to original values
                    ema.resume(model)

                    # Log to terminal
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    for k, v in results.items():
                        tboard.add_scalar(f'dev/{k}', v, step)

                    util.visualize(tboard,args.dev_eval_file, step=step, split='dev', num_visuals=args.num_visuals)
def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            log_p1, log_p2 = model(cw_idxs, qw_idxs)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())
