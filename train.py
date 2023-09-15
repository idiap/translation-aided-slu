#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import gc
import pickle
from collections import defaultdict, Counter

from torch.utils.tensorboard import SummaryWriter
import os, glob
import traceback as tb

from utils import dict_send_to
from tqdm import tqdm
import time, datetime
import argparse
import json
import traceback
from hyperparams import hparams as hp
import torch
from torch import nn
import logging
from utils import infolog, checkpoint
from models import model, build
from dataloader import get_feeder, pick_eval_samples
from functools import partial
import sys
import faulthandler, signal
from datetime import timedelta
from infer import infer_batches

def main(args):
    model_dir = args.model_dir
    logdir = args.log_dir if args.log_dir is not None else model_dir

    if os.path.exists(model_dir) and os.listdir(model_dir) and args.restore_from is None:
        args.restore_from = model_dir

    if args.restore_from and os.path.isdir(args.restore_from):
        logdir = args.restore_from
        model_dir = args.restore_from
        args.restore_from = None

    time_id = datetime.datetime.now().strftime('%m%d_%H%M')

    torch.manual_seed(0)
    if args.ddp:
        from torch import distributed as dist
        dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=10))
        rank = dist.get_rank()
        local_rank = args.local_rank
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        map_location = lambda _, __:  _.cuda(local_rank)
        print("Rank: %d, Local rank: %d, World size: %d" % (rank, local_rank, world_size))
    else:
        rank = local_rank = 0
        world_size = 1
        map_location = {}

    if rank == 0 and not args.no_write:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=logdir)
    else:
        writer = None

    infolog.set_logger(os.path.join(logdir, 'outputs_%s_%d.log' % (time_id, rank)) if not args.no_write else None)
    sys.stdout = infolog.StreamToLogger(logging.root, logging.INFO)
    sys.stderr = infolog.StreamToLogger(logging.root, logging.ERROR)

    logging.info("Command: " + str(' '.join(sys.argv)))
    if os.path.exists(os.path.join(model_dir, 'hparams.json')):
        hp_ = json.load(open(os.path.join(model_dir, 'hparams.json')))
        keys = set(hp_.keys()).union(hp._hparam_types.keys())
        logging.info("Restoring hparams...")
        for k in keys:
            if k in hp_ and hp.get(k, None) != hp_.get(k, None):
                logging.info("Overriden hparam %s: %s -> %s" % (k, str(hp.get(k, None)), str(hp_.get(k, None))))
        hp.override_from_dict(hp_)
    if args.hparams and os.path.isfile(args.hparams):
        hp.override_from_dict(json.load(open(args.hparams)))
    elif args.hparams:
        hp.parse(args.hparams)

    if os.path.exists(os.path.join(model_dir, 'args.json')):
        args_ = json.load(open(os.path.join(model_dir, 'args.json')))
        args_c = vars(args)
        keys = set(args_.keys()).union(args_c.keys())
        logging.info("Found args from previous run...")
        for k in keys:
            if args_.get(k, '') != args_c.get(k, ''):
                logging.info("Changed arg %s: %s -> %s" % (k, str(args_.get(k, '')), str(args_c.get(k, ''))))

    if os.path.exists(os.path.join(model_dir, 'best_metrics.json')):
        best_metrics = json.load(open(os.path.join(model_dir, 'best_metrics.json')))
        best_metrics['history'] = defaultdict(list, best_metrics.get('history', {}))
    else:
        best_metrics = {'history': defaultdict(list)}


    if not torch.cuda.is_available():
        map_location = lambda _, __:  _.cpu()

    if rank == 0:
        values = hp.values()
        logging.info('Hyperparameters:\n' + '\n'.join(['  %s: %s' % (name, values[name]) for name in sorted(values)]))

        if not args.no_write:
            if os.path.exists(os.path.join(model_dir, 'hparams.json')):
                os.rename(os.path.join(model_dir, 'hparams.json'),
                          os.path.join(model_dir, 'hparams.json.' + time_id))
            if os.path.exists(os.path.join(model_dir, 'args.json')):
                os.rename(os.path.join(model_dir, 'args.json'),
                          os.path.join(model_dir, 'args.json.' + time_id))
            open(os.path.join(logdir, 'hparams.json'), 'w').write(hp.to_json(indent=1))
            open(os.path.join(logdir, 'args.json'), 'w').write(json.dumps(vars(args), indent=1))


    if args.eval_steps is not None:
        eval_steps = [int(s) for s in args.eval_steps.split(':')]
    else:
        eval_steps = None

    feeder, feeder_eval, processor = get_feeder(args, hp, rank, world_size, get_eval=rank == 0)

    logging.info("Using %d GPUs" % torch.cuda.device_count())
    m = model.Model(hp)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m.to(device)
    if args.ddp:
        example_param = list(m.parameters())[5]
        logging.info("Model on %s" % str(example_param.device))
        m = nn.parallel.DistributedDataParallel(m, device_ids=[local_rank], output_device=local_rank)
    else:
        m = nn.DataParallel(m)

    wd, nwd = [], []
    for name, param in m.named_parameters():
        if model.is_weight_decayed(name):
            wd.append(param)
        else:
            nwd.append(param)

    optim = torch.optim.AdamW([{'params': wd, 'weight_decay': hp.reg_weight}, {'params': nwd, 'weight_decay': 0.}],
                              lr=hp.max_lr, eps=hp.adam_eps, betas=(0.9, 0.999))

    # optim = torch.optim.AdamW(m.parameters(),
    #                           lr=hp.max_lr, eps=hp.adam_eps, betas=(0.9, 0.999), weight_decay=hp.reg_weight)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=partial(model.learning_rate_schedule, hp=hp))

    global_step = None
    ckpt_path = checkpoint.find_ckpt(model_dir)
    if ckpt_path:
        global_step = checkpoint.load_model(ckpt_path, m, optim, sched, map_location)
        logging.info("Restore from previous run at " + model_dir + " from " + ckpt_path + ", step %d" % global_step)
    elif args.restore_from:
        global_step = checkpoint.load_model(args.restore_from, m, optim, sched, map_location, args.reset_training)
        logging.info("Restore from " + args.restore_from + ", step %s" % str(global_step))
    if global_step is None:
        global_step = 0
    if os.path.exists(os.path.join(logdir, 'feeder_%d_%d.pth' % (global_step, rank))):
        feeder_path = os.path.join(logdir, 'feeder_%d_%d.pth' % (global_step, rank))
    else:
        logging.warn("No feeder at step %d of rank %d found, try to use latest..." % (global_step, rank))
        feeder_path = checkpoint.find_ckpt(logdir, prefix='feeder_', suffix='_%d.pth' % rank)
    if feeder_path:
        logging.info("Load feeder state from " + feeder_path)
        feeder.load_state_dict(torch.load(feeder_path))

    if max(hp.warmup_steps, hp.data_warmup_steps, hp.freeze_steps) > 3 * min(hp.warmup_steps, hp.data_warmup_steps, hp.freeze_steps):
        logging.warn("Warmup steps=%d, data warmup steps=%d, freeze steps=%d, may not be reasonable" % (hp.warmup_steps, hp.data_warmup_steps, hp.freeze_steps))

    feeder.global_step = global_step
    feeder.daemon = True
    feeder.start()

    if args.reference_model:
        reference = torch.load(args.reference_model, map_location=map_location)
        reference_model = reference['model']
        if hp.use_fisher_l2sp:
            from models.model import compute_fisher_weights_from_adam
            l2_sp_weights = compute_fisher_weights_from_adam(reference, m, hp.l2_sp_weight)
            for k, v in l2_sp_weights.items():
                l2_sp_weights[k] = v.to(device)
        for k, v in reference_model.items():
            reference_model[k] = v.to(device)

    m.train()

    time_window = infolog.ValueWindow(100)
    loss_window = infolog.ValueWindow(100)
    recent_fails = infolog.ValueWindow(10)
    summary_windows = [('time', time_window), ('loss', loss_window)]

    def is_lna(name):
        return 'attention' in name or 'layer_norm' in name or '_attn' in name \
               or 'embed_positions' in name or 'layernorm' in name or '.adaptor' in name

    if global_step < hp.freeze_steps:
        model.freeze_module(m, hp.freeze_module, keep_encoder_frozen=hp.freeze_feature_encoder)
        if hp.use_lna:
            model.freeze_module(m, prefix='', frozen=True, keep_encoder_frozen=hp.freeze_feature_encoder)
            model.freeze_module(m, name_fn=is_lna, frozen=False, keep_encoder_frozen=hp.freeze_feature_encoder)
    else:
        if hp.use_lna:
            model.freeze_module(m, name_fn=is_lna, frozen=False, keep_encoder_frozen=hp.freeze_feature_encoder)
        model.freeze_module(m, hp.freeze_module, frozen=False, keep_encoder_frozen=hp.freeze_feature_encoder)

    if rank == 0:
        n_param = 0
        for name, param in m.named_parameters():
            logging.info("%s %s %s" % (name, param.shape, param.requires_grad))
            if param.requires_grad:
                n_param += param.numel()
        logging.info("Total number of trainable parameters: %d" % n_param)

    model.init_module(m, hp.reinit_module)

    logging.info("Start training run")
    optim.zero_grad()
    accum = args.accumulation_steps

    length_window = infolog.ValueWindow(100)
    sample_window = infolog.ValueWindow(100)
    summary_windows += [('length', length_window), ('sample', sample_window)]
    if hp.use_decoder:
        token_window = infolog.ValueWindow(100)
        summary_windows += [('token', token_window)]

    def signal_handler(sig, frame):
        if not args.no_write:
            logging.info("Got signal %d, saving and exiting" % sig)
            step = feeder.global_step
            checkpoint.save_model(os.path.join(model_dir, 'model.ckpt-%d' % step), m, optim, sched)
            torch.save(feeder.state_dict(), os.path.join(logdir, 'feeder_%d_%d.pth' % (global_step, rank)))
        sys.exit(0)

    if not args.disable_auto_save:
        signal.signal(signal.SIGINT, signal_handler)

    accum_loss_keys = ['loss']
    if hp.ctc_weight > 0:
        accum_loss_keys.append('ctc_loss')
    if hp.use_decoder:
        accum_loss_keys.append('decoder_loss')
    if hp.use_language_adversarial:
        accum_loss_keys.append('language_adversarial_loss')
    if hp.use_classifier:
        accum_loss_keys.append('classifier_loss')
    if hp.l2_sp_weight > 0:
        accum_loss_keys.append('l2_sp_loss')
    accum_keys = accum_loss_keys + ['n_frames', 'batch_size', 'n_decoder_tokens']
    if hp.use_language_adversarial:
        accum_keys.append('language_adversarial_acc')
        lang_adv_acc_window = infolog.ValueWindow(100)
        summary_windows.append(('language_adversarial_acc', lang_adv_acc_window))

    while True:
        tic = time.time()
        accum_losses = defaultdict(float)
        src_langs = []

        batches = defaultdict(list)

        for i in range(accum):
            batch = feeder.get_batch()
            if batch is None:
                batch = feeder.get_batch()
            batches[batch['type']].append(batch)

        for batch_type, batch_list in batches.items():
            n_samples = 0
            for batch in batch_list:
                if batch['type'] == 's2s' and hp.loss_normalize_type == 'sample':
                    n_samples += (batch['decoder_label_lengths' if hp.use_decoder else 'label_lengths']).sum().item()
                else:
                    n_samples += len(batch['inputs'])
                if hp.use_language_adversarial:
                    n_samples += len(batch['inputs'])

            type_accum_losses = defaultdict(float)

            for batch in batch_list:
                # logging.info("%s %.2E %.2E" % (str(batch['inputs'].shape), batch['inputs'].shape[0] * batch['inputs'].shape[1],
                #                            batch['inputs'].shape[0] * batch['inputs'].shape[1] * batch['inputs'].shape[1]))
                # logging.info("%s (%.2f) %d (%.2f) %d" % (str(batch['inputs'].shape), sample_window.average,
                #                                          batch['input_lengths'].sum(), length_window.average,
                #                                          batch['label_lengths'].sum(), label_length_window.average))
                # length_window.append(batch['input_lengths'].sum())
                # label_length_window.append(batch['label_lengths'].sum())
                # sample_window.append(batch['inputs'].shape[0])
                batch = dict_send_to(batch, device)
                oom = outputs = losses = False
                try:
                    outputs = m(**batch)
                    losses = model.compute_loss(batch, outputs, m.module, hp, global_step)
                    (losses['loss'] / n_samples / len(batches)).backward()

                    for key in accum_keys:
                        if key in losses:
                            type_accum_losses[key] += \
                                losses[key].item() if torch.is_tensor(losses[key]) else losses[key]
                    src_langs.extend(batch['src_lang'].detach().cpu().numpy().tolist())
                except Exception as e:
                    logging.error("Failed due to %s, input shape: %s, target shape: %s" %
                                  (type(e).__name__, str(batch['inputs'].shape), str(batch['labels'].shape)))
                    traceback.print_exc()
                    if len(recent_fails._values) == 10 and recent_fails._values[0] > global_step - 20:
                        logging.error("Too many failures, exiting")
                        return
                    recent_fails.append(global_step)
                    optim.zero_grad()
                    oom = True
                del losses
                del outputs
                if oom:
                    gc.collect()
                    if True or args.ddp or torch.cuda.memory_allocated() > torch.cuda.max_memory_allocated() * 0.9:
                        logging.info("Current memory: %d, cf. peak %d" %
                                     (torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()))
                        if not args.no_write:
                            if rank == 0:
                                checkpoint.cleanup_checkpoint(model_dir)
                                checkpoint.save_model(model_dir, m, optim, sched, global_step)
                            torch.save(feeder.state_dict(), os.path.join(logdir, 'feeder_%d_%d.pth' % (global_step, rank)))
                            if rank != 0:
                                time.sleep(20)
                        sys.exit(1)
                    continue

            for key in type_accum_losses:
                accum_losses[key] += type_accum_losses[key] / (n_samples if key in accum_loss_keys else 1)

        grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), hp.max_grad_norm)
        optim.step()
        optim.zero_grad()
        sched.step()
        global_step += 1
        feeder.global_step = global_step
        if hp.use_language_adversarial:
            accum_losses['language_adversarial_acc'] /= len(src_langs)
            lang_adv_acc_window.append(accum_losses['language_adversarial_acc'])
            counter = Counter()
            counter.update(src_langs)
            accum_losses['language_adversarial_base'] = counter.most_common(1)[0][1] / len(src_langs)

        if hp.l2_sp_weight > 0:
            l2_sp_loss = model.regularize_l2_sp(m, reference_model,
                                                l2_sp_weights if hp.use_fisher_l2sp else hp.l2_sp_weight)
            accum_losses['l2_sp_loss'] = l2_sp_loss.item()

        if global_step == hp.freeze_steps:
            if hp.use_lna:
                model.freeze_module(m, name_fn=is_lna, frozen=False, keep_encoder_frozen=hp.freeze_feature_encoder)
            else:
                model.freeze_module(m, hp.freeze_module, frozen=False, keep_encoder_frozen=hp.freeze_feature_encoder)

        if rank == 0:
            losses = dict_send_to(accum_losses, torch.device('cpu'), detach=True)
            dur = time.time() - tic
            time_window.append(dur)
            loss_window.append(losses['loss'])
            sample_window.append(losses['batch_size'])
            length_window.append(losses['n_frames'])
            loss_message = ', '.join([('%s=%.4f' % (k, losses[k]) if losses[k] > 0.01 else '%s=%.3E' % (k, losses[k])) for k in accum_loss_keys if k in losses])
            if hp.use_language_adversarial:
                loss_message += ', lang_adv_acc=%.4f (%.4f/%.4f)' % (losses['language_adversarial_acc'], lang_adv_acc_window.average, losses['language_adversarial_base'])
            message = '[Step %d] %.3f sec/step (%.3f), lr=%.04E (Ave. loss %.5f), %s, %d (%.2f) samples' % (
                global_step, dur, time_window.average, sched.get_last_lr()[-1],
                loss_window.average, loss_message, losses['batch_size'], sample_window.average)

            if len(batches) > 1:
                message += ' (' + ', '.join(['%d/%d %s' % (len(v), sum([len(_['names']) for _ in v]), k) for k, v in batches.items()]) + ')'

            if hp.use_decoder and 'n_decoder_tokens' in losses:
                token_window.append(losses['n_decoder_tokens'])
                label_cnt = (losses['n_decoder_tokens'], token_window.average)
                message += ", %d (%.2f) tokens" % label_cnt
            message += ", g-norm %.2f" % grad_norm
            logging.info(message)

            if global_step % args.checkpoint_interval == 0 and not args.no_write:
                checkpoint.save_model(model_dir, m, optim, sched, global_step)
                torch.save(feeder.state_dict(), os.path.join(logdir, 'feeder_%d_%d.pth' % (global_step, rank)))
                logging.info("Save checkpoint to " + model_dir)
                checkpoint.cleanup_checkpoint(model_dir)

            if global_step % args.summary_interval == 0 and writer is not None:
                for key in accum_keys:
                    if key in losses:
                        writer.add_scalar('losses/' + key, losses[key], global_step=global_step)
                writer.add_scalar('lr', sched.get_last_lr()[-1], global_step=global_step)
                writer.add_scalar('gram_norm', grad_norm, global_step=global_step)
                for name, window in summary_windows:
                    writer.add_scalar('average/' + name, window.average, global_step=global_step)
                writer.add_scalar('memory/cuda', torch.cuda.memory_allocated(), global_step=global_step)
                writer.flush()

            if (eval_steps and global_step in eval_steps) or \
                    (eval_steps is None and
                     ((global_step % args.checkpoint_interval == 0) or
                      (global_step % args.eval_interval == 0))):
                eval_path = os.path.join(logdir, 'eval_%d' % (global_step))
                batches = feeder_eval.fetch_data()

                picked_batches = pick_eval_samples(batches, hp)

                metrics = infer_batches(m, picked_batches, eval_path, hp, device, processor,
                                        write_output=not args.no_write)

                if not args.no_write:
                    updated_bests = []
                    for k, v in metrics.items():
                        writer.add_scalar('eval/%s' % k, v, global_step=global_step)
                        if 'bleu' in k or 'acc' in k or 'rouge' in k or 'aos' in k or 'f1' in k:
                            condition = metrics[k] >= best_metrics.get(k, (0, 0))[0]
                        else:
                            condition = metrics[k] <= best_metrics.get(k, (1e9, 0))[0]
                        if condition:
                            updated_bests.append(k)
                            best_metrics['history'][k].append((global_step, metrics[k]))
                    if updated_bests:
                        name = 'model.ckpt-%d' % global_step
                        for k in updated_bests:
                            best_metrics[k] = (metrics[k], global_step)

                        json.dump(best_metrics, open(os.path.join(logdir, 'best_metrics.json'), 'w'), indent=1)

                        if global_step >= hp.data_warmup_steps and not os.path.exists(os.path.join(model_dir, name)):
                            checkpoint.save_model(model_dir, m, optim, sched, global_step, name=name)
                            torch.save(feeder.state_dict(), os.path.join(logdir, 'feeder_%d_%d.pth' % (global_step, rank)))
                            logging.info("Save best model to " + name)
                            checkpoint.cleanup_checkpoint(model_dir)


                m.train()
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True,
                        help="Directory to save checkpoints and resume (when --restore_from is not specified)")
    parser.add_argument('--log-dir', default=None, help="Directory to save log and tfevents")
    parser.add_argument('--data-dir', required=True, help="Directory with data and metadata")
    parser.add_argument('--vocab-path', type=str, default=None, help="Path to vocab.json")
    parser.add_argument('--train_meta', type=str, default=None,
                        help="Metadata file for training, use metadata.train.txt under data-dir when not given")
    parser.add_argument('--eval_meta', type=str, default=None,
                        help="Metadata file for eval, use metadata.eval.txt under data-dir when not given")
    parser.add_argument('--eval_steps', type=str, default=None,
                        help="Steps of checkpoints to run eval on. Run on all steps when not specified")
    parser.add_argument('--src_lang', type=str, default='',
                        help="Source languages")
    parser.add_argument('--tgt_lang', type=str, default='',
                        help="Target languages")
    parser.add_argument('--datasets', type=str, default='',
                        help="Datasets to use")
    parser.add_argument('--checkpoint_interval', type=int, default=10000)
    parser.add_argument('--eval_interval', type=int, default=2000)
    parser.add_argument('--summary_interval', type=int, default=500)
    parser.add_argument('--restore_from', help='Path of checkpoint or run directory to restore', default=None)
    parser.add_argument('--reference_model', help='Path of checkpoint as reference model', default=None)
    parser.add_argument('--hparams', default='', help='Alternative hparams')
    parser.add_argument('--ddp', help='Using DDP', action='store_true')
    parser.add_argument('--max_retry', help='Number of max retry', type=int, default=0)
    parser.add_argument('--max_steps', help='Number of max steps', type=int, default=-1)
    parser.add_argument('--accumulation_steps', help='Number of steps for gradient accumulation', type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--rewrite_output_dir", action='store_true', default=False)
    parser.add_argument("--no_write", action='store_true', help="Prevent from writing any files", default=False)
    parser.add_argument("--reset_training", action='store_true', default=False)
    parser.add_argument("--max_epoch", type=int, default=0)
    parser.add_argument("--disable_auto_save",  action='store_true', default=False)

    args, unparsed = parser.parse_known_args()
    print('unparsed:', unparsed)


    if args.rewrite_output_dir and os.path.exists(args.model_dir):
        logging.info('Removing existing model dir %s' % args.model_dir)
        os.system('rm -rf %s' % args.model_dir)

    if args.max_retry == 0:
        try:
            main(args)
        except:
            tb.print_exc()
    else:
        import multiprocessing as mp
        from multiprocessing import Process
        mp.set_start_method('spawn')
        short_retry_cnt = 0
        for i in range(args.max_retry + 1):
            tic = time.time()
            if i != 0:
                print("\n==========Retry %d==========\n" % i)
            p = Process(target=main, args=(args,))
            p.start()
            p.join()
            if p.exitcode == 0:
                print("Success")
                i = None
                break
            if time.time() - tic < 600:
                short_retry_cnt += 1
            else:
                short_retry_cnt = 0
            if short_retry_cnt == 3:
                print("Too many short retries, abort")
                break
        if i is not None:
            print("Max retry reached...")