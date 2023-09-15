#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import os, logging, time, traceback
from utils import dict_send_to
import pickle
from models.model import compute_loss, sample_to_frame, frame_to_sample
import jiwer, evaluate
import json
import torch
import argparse
import datetime
import sys
from utils import infolog, checkpoint
from models import model, build
from dataloader import get_feeder
from hyperparams import hparams as hp
from utils.languages import id_to_lang
from functools import partial
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter

metrics = None
metrics_type = {}

def load_metrics():
    global metrics, metrics_type
    bleu = evaluate.load("bleu", keep_in_memory=True)
    rouge = evaluate.load("rouge", keep_in_memory=True)

    def compute_rouge(labels, preds, key):
        results = rouge.compute(predictions=preds, references=labels)
        return results[key]

    def compute_bleu(labels, preds):
        labels = [[l] for l in labels]
        results = bleu.compute(predictions=preds, references=labels)
        return results['bleu']

    all_metrics = {'wer': jiwer.wer, 'cer': jiwer.cer, 'bleu': compute_bleu}
    metrics = dict([(k, all_metrics[k]) for k in hp.eval_metrics if k in all_metrics])

    if 'rouge' in hp.eval_metrics:
        for ch in '12L':
            metrics['rouge' + ch] = partial(compute_rouge, key='rouge' + ch)
            metrics_type['rouge' + ch] = 'sum'

    def normalized_error_rate(labels, preds, fn):
        labels = [''.join([c for c in l if c.isalpha() or c in " '"]).lower() for l in labels]
        preds = [''.join([c for c in p if c.isalpha() or c in " '"]).lower() for p in preds]
        return fn(labels, preds)

    for ex_metric in ['wer', 'cer']:
        if ex_metric in metrics:
            metrics['norm_' + ex_metric] = partial(normalized_error_rate, fn=metrics[ex_metric])

    def accuracy(labels, preds, component=None):
        if component is not None:
            return np.mean([l[component] == p[component] for l, p in zip(labels, preds)])
        return np.mean([tuple(l) == tuple(p) for l, p in zip(labels, preds)])

    for i in range(hp.classifier_num_targets):
        metrics['acc_' + str(i)] = partial(accuracy, component=i)
        metrics_type['acc_' + str(i)] = 'cls'
    metrics['acc'] = accuracy
    metrics_type['acc'] = 'cls'

def compute_qa_metrics(sample_outputs, hp):
    def compute_aos(label, pred):
        if label[1] == 0 or pred[1] == 0:
            return 0
        assert label[0] <= label[1]
        assert pred[0] <= pred[1]
        inter = (max(label[0], pred[0]), min(label[1], pred[1]))
        union = (min(label[0], pred[0]), max(label[1], pred[1]))
        return max(inter[1] - inter[0], 0) / (union[1] - union[0])

    question_segments = defaultdict(list)
    question_answers = {}
    aos_scores = []
    result = {}
    for sample_output in sample_outputs:
        if hp.qa_label_type == 'sample':
            q, c = sample_output['name'].split(',')
            assert (q, c) not in question_answers
            c = c[:-len(c.split('-')[-1]) - 1]
            question_segments[(q, c)].append(sample_output)
            if len(sample_output['qa_label_sample']) == 0:
                continue
            aos_score = max([compute_aos(label, sample_output['qa_pred_sample']) for label in sample_output['qa_label_sample']])
            aos_scores.append(aos_score)
            question_answers[(q, c)] = (sample_output['name'], {'aos': aos_score})
        else:
            aos_score = max(
                [compute_aos(label, sample_output['qa_pred']) for label in sample_output['qa_label']])
            aos_scores.append(aos_score)

    result['aos'] = np.mean(aos_scores)

    if hp.qa_label_type == 'sample':
        group_aos_scores = []
        segment_acc = []
        for q, c in question_segments:
            # assert len(question_segments[(q, c)]) == max([int(s['name'].split('-')[-1]) + 1 for s in question_segments[(q, c)]])
            max_prob = -1e9
            best_i = 0
            for i, sample_output in enumerate(question_segments[(q, c)]):
                if hp.qa_segment_cls:
                    if sample_output['qa_segment_score'] > max_prob:
                        max_prob = sample_output['qa_segment_score']
                        best_i = i
                else:
                    if sample_output['qa_pred_sample'][1] != 0:
                        if sample_output['qa_score'] > max_prob:
                            max_prob = sample_output['qa_score']
                            best_i = i
            if max_prob == -1e9:
                group_aos_scores.append(0)
                segment_acc.append(0)
                continue
            sample_output = question_segments[(q, c)][best_i]
            if sample_output['name'] != question_answers[(q, c)][0]:
                group_aos_scores.append(0)
                segment_acc.append(0)
                continue
            segment_acc.append(1)
            group_aos_scores.append(question_answers[(q, c)][1]['aos'])
        result.update({'group_aos': np.mean(group_aos_scores), 'segment_acc': np.mean(segment_acc)})
    return result

def infer_batches(model, batches, eval_path, hp, device='cpu', processor=None, write_output=True):
    os.makedirs(eval_path, exist_ok=True)
    model.eval()
    if hasattr(model, 'module'):
        eval_model = model.module
    else:
        eval_model = model
    if not metrics:
        load_metrics()

    logging.info('Running %d evals, to %s' % (len(batches), eval_path))

    if processor:
        tokenizer = processor['tokenizer']
        decoder_tokenizer = processor['decoder_tokenizer']

    infer_outputs = []
    start_tic = time.time()
    has_ctc_label = has_dec_label = has_cls_label = False
    has_qa_samples = False
    for i, batch in enumerate(batches):
        try:
            eval_tic = time.time()
            batch_ = dict_send_to(batch, device)
            with torch.no_grad():
                outputs = eval_model(**batch_, generate=True, num_beams=hp.eval_num_beams,
                                     length_penalty=hp.eval_length_penalty)
            losses = None
            if 'labels' in batch or 'decoder_labels' in batch:
                losses = compute_loss(batch_, outputs, eval_model, hp)
                losses = dict_send_to(losses, 'cpu', detach=True, as_numpy=True)
            outputs = dict_send_to(outputs, 'cpu', detach=True, as_numpy=True)
            batch = dict_send_to(batch, 'cpu', detach=True, as_numpy=True)

            for j in range(len(batch['names'])):
                sample_output = {}
                sample_output['name'] = batch['names'][j]
                sample_output['subset'] = batch['dataset_names'][j] + '/' + id_to_lang[batch['src_lang'][j].item()] \
                                           + '_' + id_to_lang[batch['tgt_lang'][j].item()]
                if processor and batch['type'] == 's2s':
                    if hp.ctc_weight > 0:
                        sample_output['ctc_logit'] = outputs['logits'][j]
                        pred = tokenizer.decode(outputs['ctc_logit'][j].argmax(axis=-1))
                        sample_output['ctc_pred'] = pred
                    if 'decoder_outputs' in outputs:
                        sample_output['decoder_pred'] = decoder_tokenizer.decode(
                            outputs['decoder_outputs'][j], skip_special_tokens=True)
                if batch['type'] == 'cls':
                    if isinstance(outputs['classifier_outputs'], dict):
                        if hp.classifier_head_type == 'qa':
                            sample_output['classifier_logit'] = outputs['classifier_outputs']['logits'][j]
                            if hp.qa_label_type == 'sample':
                                segment_start = sample_to_frame(batch['input_segments'][j], batch['input_lengths'][j],
                                                                outputs['classifier_outputs']['lengths'][j],
                                                                outputs['decoder_mask'][j].sum(-1))
                            else:
                                segment_start = batch['input_segments'][j]
                            segment_end = outputs['classifier_outputs']['lengths'][j]
                            max_prob = -1e9
                            best_pair = None
                            has_qa_samples = True
                            for start_idx in range(segment_start, segment_end):
                                for end_idx in range(start_idx, segment_end):
                                    if sample_output['classifier_logit'][start_idx][0] + \
                                            sample_output['classifier_logit'][end_idx][1] > max_prob:
                                        max_prob = sample_output['classifier_logit'][start_idx][0] + \
                                                   sample_output['classifier_logit'][end_idx][1]
                                        best_pair = (start_idx, end_idx)

                            sample_output['qa_cand_range'] = (segment_start, segment_end - 1)
                            sample_output['qa_cand_range_sample'] = (
                            batch['input_segments'][j], batch['input_lengths'][j] - 1)
                            if hp.qa_label_type == 'sample':
                                if not hp.qa_segment_cls:
                                    if max_prob < sample_output['classifier_logit'][0].sum():
                                        best_pair = (segment_start, segment_start)
                                        max_prob = sample_output['classifier_logit'][0].sum()
                                best_pair_sample = tuple([frame_to_sample(t, batch['input_lengths'][j],
                                                                          outputs['classifier_outputs']['lengths'][j],
                                                                          outputs['decoder_mask'][j].sum(-1))
                                                          for t in best_pair])
                                sample_output['qa_label_sample'] = np.asarray([t for t in batch['labels'][j].reshape([-1, 2]) if t[0] > 0])
                                sample_output['qa_label'] = np.asarray([
                                    sample_to_frame(t, batch['input_lengths'][j],
                                                    outputs['classifier_outputs']['lengths'][j],
                                                    outputs['decoder_mask'][j].sum(-1)) for t in batch['labels'][j]]).\
                                    reshape([-1, 2])
                                sample_output['qa_label'] = sample_output['qa_label'][sample_output['qa_label'][:, 0] > 0]
                                if hp.qa_segment_cls:
                                    sample_output['qa_segment_score'] = outputs['classifier_outputs']['segment_logits'][j][0]
                            else:
                                best_pair_sample = best_pair
                                sample_output['qa_label'] = sample_output['qa_label_sample'] = np.asarray(
                                    [t for t in batch['labels'][j].reshape([-1, 2]) if t[0] > 0])

                            sample_output['qa_score'] = max_prob
                            sample_output['qa_pred_sample'] = best_pair_sample
                            sample_output['classifier_pred'] = best_pair_sample
                            if hp.qa_segment_cls:
                                sample_output['classifier_pred'] += (sample_output['qa_segment_score'].item(),)
                        else:
                            sample_output['classifier_logit'] = outputs['classifier_outputs']['logits'][:, j]
                            sample_output['classifier_pred'] = sample_output['classifier_logit'].argmax(axis=-1)
                    else:
                        sample_output['classifier_pred'] = outputs['classifier_outputs'][j]

                if batch['type'] == 's2s':
                    if hp.ctc_weight > 0 and 'labels' in batch:
                        has_ctc_label = True
                        label = batch['labels'][j]
                        label[label == -100] = decoder_tokenizer.pad_token_id
                        sample_output['ctc_label'] = tokenizer.decode(label, group_tokens=False)
                    if 'decoder_labels' in batch:
                        has_dec_label = True
                        # sample_output['decoder_loss'] = losses['decoder_losses'][j]
                        label = batch['decoder_labels'][j]
                        label[label == -100] = decoder_tokenizer.pad_token_id
                        sample_output['decoder_label'] = decoder_tokenizer.decode(
                            label, skip_special_tokens=True)
                if batch['type'] == 'cls' and 'labels' in batch:
                    has_cls_label = True
                    if 'classifier_losses' in losses:
                        if hp.classifier_head_type == 'qa':
                            sample_output['classifier_loss'] = losses['classifier_losses'][j]
                        else:
                            sample_output['classifier_loss'] = losses['classifier_losses'][:, j]
                    if hp.classifier_head_type == 'qa':
                        sample_output['classifier_label'] = batch['labels'][j]
                    else:
                        sample_output['classifier_label'] = batch['labels'][j][:hp.classifier_num_targets]
                # if hp.use_language_adversarial:
                #     sample_output['lang_adversarial_loss'] = losses['lang_adversarial_losses'][j]
                #     sample_output['lang_adversarial_logit'] = outputs['lang_adversarial_outputs']['logits'][j]
                #     sample_output['lang_adversarial_pred'] = sample_output['lang_adversarial_logit'].argmax(axis=-1)
                #     sample_output['lang_adversarial_label'] = batch['src_lang'][j]
                infer_outputs.append(sample_output)

            logging.info('Finished batch %d/%d in %.2f sec, samples: %s' % (
                i, len(batches), time.time() - eval_tic, batch['names']))
        except:
            traceback.print_exc()

    logging.info("Total %d batches of %d samples, cost %.2f sec" %
                 (len(batches), len(infer_outputs), time.time() - start_tic))

    def get_col(samples, key, idx=None):
        if idx is not None:
            return [samples[i][key] for i in idx if key in samples[i]]
        return [s[key] for s in samples if key in s]

    return_metrics = {}
    if has_ctc_label or has_dec_label or has_cls_label:
        if has_cls_label:
            t = get_col(infer_outputs, 'classifier_loss')
            if t:
                return_metrics['classifier_loss'] = np.mean(t).item()

        all_subsets = get_col(infer_outputs, 'subset')
        all_subset_set = set(all_subsets)

        langs = [l['subset'].split('/')[1].split('_') for l in infer_outputs]
        all_subset_idx = [('asr', [i for i, s in enumerate(infer_outputs) if langs[i][0] == langs[i][1]]),
                          ('st', [i for i, s in enumerate(infer_outputs) if
                                  langs[i][0] != langs[i][1] and langs[i][1] not in ['u', 'tag'] and langs[i][1][-4:] != '-sum'])]
        if len(set([langs[i][0] for i, s in enumerate(infer_outputs) if langs[i][1] == 'u'])) > 1:
            all_subset_idx.append(('cls', [i for i, s in enumerate(infer_outputs) if langs[i][1] == 'u']))

        for subset in all_subset_set:
            subset_idx = [i for i, s in enumerate(infer_outputs) if s['subset'] == subset]
            all_subset_idx.append((subset, subset_idx))

        for subset, subset_idx in all_subset_idx:
            if not subset_idx:
                continue
            targets = []
            if has_ctc_label:
                targets.append(('', get_col(infer_outputs, 'ctc_label', subset_idx),
                                get_col(infer_outputs, 'ctc_pred', subset_idx)))
            if has_dec_label:
                targets.append(('decoder', get_col(infer_outputs, 'decoder_label', subset_idx),
                                get_col(infer_outputs, 'decoder_pred', subset_idx)))
            if has_cls_label:
                targets.append(('classifier', get_col(infer_outputs, 'classifier_label', subset_idx),
                                get_col(infer_outputs, 'classifier_pred', subset_idx)))

            if has_qa_samples:
                qa_samples = [infer_outputs[i] for i in subset_idx if 'qa_score' in infer_outputs[i]]
                if qa_samples:
                    qa_metrics = compute_qa_metrics(qa_samples, hp)
                    for k, v in qa_metrics.items():
                        return_metrics['%s/%s' % (subset, k)] = v
                        logging.info("%s/%s: %.4f" % (subset, k, v))

            for prefix, labels, preds in targets:
                if labels and preds:
                    for key, fn in metrics.items():
                        if (prefix == 'classifier') != (metrics_type.get(key, 's2s') == 'cls'):
                            continue
                        if (metrics_type.get(key, 's2s') == 'sum') != (subset.endswith('-sum')):
                            continue

                        r = fn(labels, preds)
                        m_key = prefix + '_' + key if prefix else key
                        m_key = subset + '/' + m_key if subset else m_key

                        logging.info("%s: %.4f" % (m_key, r))
                        return_metrics[m_key] = r
    # if hp.use_language_adversarial:
    #     return_metrics['lang_adversarial_loss'] = np.mean(get_col(infer_outputs, 'lang_adversarial_loss')).item()
    #     return_metrics['lang_adversarial_acc'] = metrics['acc'](
    #         get_col(infer_outputs, 'lang_adversarial_label'),
    #         get_col(infer_outputs, 'lang_adversarial_pred'))

    if write_output:
        fw = open(os.path.join(eval_path, 'preds.jsonl'), 'w', encoding='utf-8')
        infer_outputs.sort(key=lambda x: x['name'])
        for i in range(len(infer_outputs)):
            r = {'name': infer_outputs[i]['name'], 'subset': infer_outputs[i]['subset']}
            for key in ['decoder_label', 'decoder_pred', 'pred', 'classifier_pred', 'classifier_label']:
                if key in infer_outputs[i]:
                    r[key] = infer_outputs[i][key]
                    if isinstance(r[key], np.ndarray):
                        r[key] = r[key].tolist()
            fw.write(json.dumps(r, ensure_ascii=False) + '\n')

        pickle.dump(infer_outputs, open(os.path.join(eval_path, 'logits.pkl'), 'wb'))
    return return_metrics


def main(args):
    if os.path.isdir(args.model_path):
        model_dir = args.model_path
        model_paths = checkpoint.find_ckpt(model_dir, True)
    else:
        model_dir = os.path.dirname(args.model_path)
        model_paths = [(None, args.model_path)]
    logdir = args.output_path if args.output_path is not None else model_dir
    time_id = datetime.datetime.now().strftime('%m%d_%H%M')
    os.makedirs(logdir, exist_ok=True)
    exclude_steps = args.exclude_steps.split(':')
    include_steps = args.include_steps.split(':') if args.include_steps else None

    infolog.set_logger(os.path.join(logdir, 'outputs_eval_%s.log' % (time_id)))
    sys.stdout = infolog.StreamToLogger(logging.root, logging.INFO)
    sys.stderr = infolog.StreamToLogger(logging.root, logging.ERROR)

    logging.info("Command: " + str(' '.join(sys.argv)))
    if os.path.exists(os.path.join(model_dir, 'hparams.json')):
        hp_ = json.load(open(os.path.join(model_dir, 'hparams.json')))
        keys = set(hp_.keys()).union(hp._hparam_types.keys())
        logging.info("Restoring hparams...")
        for k in keys:
            if hp.get(k, None) != hp_.get(k, None):
                logging.info("Different hparam %s: %s -> %s" % (k, str(hp.get(k, None)), str(hp_.get(k, None))))
        hp.override_from_dict(hp_)
    if args.hparams and os.path.isfile(args.hparams):
        hp.override_from_dict(json.load(open(args.hparams)))
    else:
        hp.parse(args.hparams)

    _, feeder_eval, processor = get_feeder(args, hp, 0, 1, get_train=False, get_eval=True, shuffle_eval=False,
                                           reduce_eval_batch=False)

    m = model.Model(hp)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m.to(device)

    if not torch.cuda.is_available():
        map_location = lambda _, __:  _.cpu()
    else:
        map_location = {}

    filtered_paths = []
    for _, model_path in model_paths:
        if include_steps and model_path.split('-')[-1] not in include_steps:
            logging.info("Skipping %s" % model_path)
            continue
        if model_path.split('-')[-1] in exclude_steps:
            logging.info("Skipping %s" % model_path)
            continue
        filtered_paths.append((_, model_path))
    model_paths = filtered_paths
    logging.info("Evaluating following models: " + '\n'.join([p for _, p in model_paths]))
    all_metrics = {}

    if args.tb_prefix:
        writer = SummaryWriter(log_dir=model_dir)


    for _, model_path in model_paths:
        global_step = checkpoint.load_model(model_path, m, None, None, map_location)
        logging.info("Restore from" + model_path + ", step %s" % str(global_step))
        out_path = os.path.join(logdir, 'eval_%d_%s' % (global_step, time_id))
        batches = feeder_eval.fetch_data()
        metrics = infer_batches(m, batches, out_path, hp, device, processor)
        all_metrics[global_step if global_step is not None else model_path] = metrics
        json.dump(metrics, open(os.path.join(out_path, 'metrics.json'), 'w'), indent=2)

        if args.tb_prefix and global_step is not None:
            for key, val in metrics.items():
                writer.add_scalar(os.path.join(args.tb_prefix, key), val, global_step=global_step)

        json.dump(all_metrics, open(os.path.join(logdir, 'eval_%s_metrics.json' % (time_id)), 'w'), indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True,
                        help="Directory or path to restore model from")
    parser.add_argument('--output-path', help="Directory or path to save results")
    parser.add_argument('--data-dir', help="Directory with data and metadata")
    parser.add_argument('--vocab-path', type=str, default=None, help="Path to vocab.json")
    parser.add_argument('--src_lang', type=str, default='',
                        help="Source languages")
    parser.add_argument('--tgt_lang', type=str, default='',
                        help="Target languages")
    parser.add_argument('--datasets', type=str, default='',
                        help="Datasets to use")
    parser.add_argument('--eval_meta', type=str, default=None,
                        help="Metadata file for eval, use metadata.eval.txt under data-dir when not given")
    parser.add_argument('--hparams', default='', help='Alternative hparams')
    parser.add_argument('--exclude_steps', default='', help='Steps to exclude')
    parser.add_argument('--include_steps', default='', help='Steps to include; overriding other options')
    parser.add_argument('--tb_prefix', default=None)

    args, unparsed = parser.parse_known_args()
    print('unparsed:', unparsed)

    main(args)