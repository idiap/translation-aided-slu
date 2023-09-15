#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import os
from collections import OrderedDict
import torch
import glob
import logging


def find_ckpt(base_dir, find_all=False, prefix='model.ckpt-', suffix=None):
    max_step = 0
    result = None
    all_result = []
    for f in glob.iglob(os.path.join(base_dir, prefix + '*')):
        if suffix and not f.endswith(suffix):
            continue
        if len(splits := os.path.split(f)[-1].split('-')) > 1 and splits[1].isdigit():
            step = int(splits[1])
        elif len(splits := os.path.split(f)[-1].split('_')) > 1 and splits[1].isdigit():
            step = int(splits[1])
        else:
            continue
        all_result.append((step, f))
        if step > max_step:
            result = f
            max_step = step
    if find_all:
        return all_result
    return result

def cleanup_checkpoint(base_dir, interval=50):
    for f in glob.iglob(os.path.join(base_dir, 'model.ckpt-*')):
        step = int(os.path.split(f)[-1].split('-')[1])
        if step % interval != 0:
            logging.info("Remove checkpoint %s" % f)
            os.remove(f)


def save_model(model_dir, model=None, optim=None, sched=None, step=None, name=None):
    state_dict = {}
    if model:
        if hasattr(model, 'module'):
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()
        state_dict['model'] = model_dict
        state_dict['param_names'] = [k for k, v in model.named_parameters()]
        state_dict['buffer_names'] = [k for k, v in model.named_buffers()]
    if optim:
        state_dict['optim'] = optim.state_dict()
    if sched:
        state_dict['sched'] = sched.state_dict()
    if step:
        state_dict['step'] = step
        model_dir = os.path.join(model_dir, name if name else 'model.ckpt-%d' % step)
    try:
        torch.save(state_dict, model_dir)
    except:
        logging.error("Failed to save model to %s" % model_dir)
        if os.path.exists(model_dir):
            os.remove(model_dir)
        while True:
            wait = input("Input T to try again, or C to continue without saving")
            if wait == 'T':
                torch.save(state_dict, model_dir)
            elif wait == 'C':
                break


def load_model(model_path, model=None, optim=None, sched=None, map_location={}, restart=False):
    state_dict = torch.load(model_path, map_location=map_location)
    if 'model' in state_dict and model:
        model_dict = state_dict['model']
        if hasattr(model, 'module'):
            model = model.module
        if restart:
            for key in model.state_dict():
                if key in model_dict and model_dict[key].shape != model.state_dict()[key].shape:
                    logging.warning("Mismatched shape for %s, skip loading" % key)
                    del model_dict[key]
        if set(model.state_dict().keys()) != set(model_dict.keys()):
            logging.warning('Model parameters do not match, loading from checkpoint anyway')
            logging.warning("Missing parameters: %s" % (set(model.state_dict().keys()) - set(model_dict.keys())))
            logging.warning("Extra parameters: %s" % (set(model_dict.keys()) - set(model.state_dict().keys())))

        model.load_state_dict(model_dict, strict=False)
    if 'optim' in state_dict and optim and not restart:
        optim.load_state_dict(state_dict['optim'])

    step = None
    if not restart:
        if 'step' in state_dict:
            step = state_dict['step']
        elif 'sched' in state_dict:
            step = state_dict['sched']['_step_count']

    if 'sched' in state_dict and sched and not restart:
        sched.load_state_dict(state_dict['sched'])
        if step:
            if step != sched.last_epoch:
                logging.warn("Step=%d, while in sched step=%d" % (step, sched.last_epoch))
        else:
            step = sched.last_epoch
    return step
