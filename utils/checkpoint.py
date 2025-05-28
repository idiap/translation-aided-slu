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
import time
import traceback as tb


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
    except Exception as e:
        logging.error("Failed to save model to %s" % model_dir)
        if isinstance(e, OSError) and 'Disk quota exceed' in e.strerror:
            while True:
                time.sleep(60)
        else:
            tb.print_exc()
        if os.path.exists(model_dir):
            os.remove(model_dir)
        while True:
            wait = input("Input T to try again, or C to continue without saving")
            if wait == 'T':
                torch.save(state_dict, model_dir)
            elif wait == 'C':
                break


def load_model(model_path, model=None, optim=None, sched=None, map_location={}, restart=False):
    try:
        state_dict = torch.load(model_path, map_location=map_location)
    except:
        raise RuntimeError("Failed to load model from %s" % model_path)

    step = None
    if not restart:
        if 'step' in state_dict:
            step = state_dict['step']
        elif 'sched' in state_dict:
            step = state_dict['sched']['_step_count']
    model_ = model
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
    model = model_

    if 'sched' in state_dict and sched and not restart:
        scm = sched.state_dict()
        for key in list(state_dict['sched'].keys()):
            if key not in ['last_epoch', '_step_count', '_last_lr']:
                state_dict['sched'][key] = scm[key]
        sched.load_state_dict(state_dict['sched'])
        if step:
            if step != sched.last_epoch + 1:
                logging.warn("Step=%d, while in sched step=%d" % (step, sched.last_epoch + 1))
        else:
            step = sched.last_epoch + 1

    if 'optim' in state_dict and optim and not restart:
        osm = optim.state_dict()
        assert len(osm['param_groups']) == len(state_dict['optim']['param_groups'])
        for i in range(len(state_dict['optim']['param_groups'])): # Optim params should be decided by its initialization, since we may load from a model with different params, e.g. lr
            pg = state_dict['optim']['param_groups'][i]
            osm['param_groups'][i]['lr'] = pg['lr']
            # for key in list(pg.keys()):
            #     if key not in ['params', 'lr']:
            #         pg[key] = osm['param_groups'][i][key]
            if sched:
                osm['param_groups'][i]['lr'] = sched.get_lr()[i]

        if 'param_names' in state_dict['optim']['param_groups'][0]:
            failed_names = []
            success_names = []
            msd = dict(model.named_parameters())
            pname_to_state = {}
            osm['state'] = OrderedDict()
            for pg in state_dict['optim']['param_groups']:
                assert len(pg['param_names']) == len(pg['params'])
                for i in range(len(pg['param_names'])):
                    pname = pg['param_names'][i]
                    old_pidx = pg['params'][i]
                    if old_pidx in state_dict['optim']['state']:
                        if msd[pname].shape != state_dict['optim']['state'][old_pidx]['exp_avg'].shape:
                            failed_names.append(pname)
                            continue
                        pname_to_state[pname] = state_dict['optim']['state'][old_pidx]
            for npg in osm['param_groups']:
                for i in range(len(npg['params'])):
                    pname = npg['param_names'][i]
                    new_pidx = npg['params'][i]
                    if pname in pname_to_state:
                        osm['state'][new_pidx] = pname_to_state[pname]
                        success_names.append(pname)
            missing_names = [t for t in msd.keys() if t not in set(success_names) and t not in set(failed_names)]
            if failed_names:
                logging.warn("Failed to load optim states for %d params: %s" % (len(failed_names), failed_names))
            if len(missing_names) > 0:
                logging.warn("Missing optim states for %d params: %s" % (len(missing_names), missing_names))
            logging.info("Successfully loaded optim states for %d params: %s" % (len(success_names), success_names))
        else:
            osm['state'] = state_dict['optim']['state']

            # np = dict(model.named_parameters())
            # pid_to_name = {id(v): k for k, v in np.items()}
            # optim_states = {}
            # failed_names = []
            # success_names = []
            # for i in range(len(state_dict['param_names'])):
            #     pname = state_dict['param_names'][i]
            #     if pname.startswith('module.'):
            #         pname = pname[len('module.'):]
            #     if i in state_dict['optim']['state']:
            #         optim_states[pname] = state_dict['optim']['state'][i]
            #     else:
            #         failed_names.append(pname)
            # assert len(osm['param_groups']) == len(optim.param_groups)
            # for i in range(len(osm['param_groups'])):
            #     assert len(osm['param_groups'][i]['params']) == len(optim.param_groups[i]['params'])
            #     for j in range(len(osm['param_groups'][i]['params'])):
            #         pid = id(optim.param_groups[i]['params'][j])
            #         if pid in pid_to_name:
            #             pname = pid_to_name[pid]
            #             if pname in optim_states:
            #                 if 'exp_avg' in optim_states[pname] and optim_states[pname]['exp_avg'].shape != optim.param_groups[i]['params'][j].shape:
            #                     logging.warn("Mismatched shape for %s (%s -> %s), skip loading" % (
            #                         pname, optim_states[pname]['exp_avg'].shape,
            #                         optim.param_groups[i]['params'][j].shape))
            #                     failed_names.append(pname)
            #                     continue
            #                 osm['state'][osm['param_groups'][i]['params'][j]] = optim_states[pname]
            #                 success_names.append(pname)
            #             else:
            #                 failed_names.append(pname)
            #         else:
            #             logging.warn("Failed to find param name for %d" % pid)
            # logging.info("Successfully loaded optim states for %d params: %s" % (len(success_names), success_names))
            # logging.info("Failed to find optim states for %d params: %s" % (len(failed_names), failed_names))

        optim.load_state_dict(osm)

    return step