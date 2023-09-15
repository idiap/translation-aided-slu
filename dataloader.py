#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import copy
import io
import logging
import threading
import queue
import traceback
from collections import defaultdict
import time
import os
from utils.languages import language_id, charset
import zipfile
from typing import List
import traceback as tb

import soundfile as sd
import numpy as np
import torch
import json
import pickle

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer

np.random.seed(0)

class ProtoFeeder(threading.Thread):
    def __init__(self, processor, hparams, rank=0, world_size=1, seed=None):
        super().__init__()
        self._tokenizer = processor['tokenizer']
        self._extractor = processor['extractor']
        self._decoder_tokenizer = processor['decoder_tokenizer']

        self.queue = queue.Queue(maxsize=128)
        self.rand = np.random.RandomState(seed if seed is not None else rank)
        self._rank = rank
        self._world_size = world_size
        self._lock = threading.Lock()

        self._hparams = hparams
        self._batch_size = hparams.batch_size
        self._batch_frame_limit = hparams.batch_frame_limit
        self._batch_quad_frame_limit = hparams.batch_quad_frame_limit

    def run(self):
        try:
            while True:
                self._enqueue_next_group()
        except Exception:
            logging.error(traceback.format_exc())

    def _enqueue_next_group(self):
        raise NotImplementedError

    def get_batch(self):
        return self.queue.get()


class Feeder(ProtoFeeder):
    def __init__(self, datadir, processor, metadata_file_path, hparams,
                 rank=0, world_size=1, shuffle=True, single=False, filter_samples=True,
                 source_lang=None, target_lang=None, seed=None, max_epoch=0):
        super(Feeder, self).__init__(processor=processor, hparams=hparams, rank=rank, world_size=world_size, seed=seed)
        self._offset = 0
        self._epoch = 0
        self.global_step = -1
        self.proto = get_input_proto(hparams)

        self._datadir = datadir

        self._shuffle = shuffle
        self._single = single
        self._filter_samples = filter_samples
        self._n_skip = 0
        self._max_epoch = max_epoch
        self.finished = False

        if hparams.data_format == 'nltLa':
            if hparams.use_infergen and hparams.infergen_mode == 'cls':
                self._cls_vocab = hparams.classifier_num_targets * \
                                  [json.load(open(os.path.join(datadir, 'categories_flat.json'), 'r'))]
            else:
                self._cls_vocab = json.load(open(os.path.join(datadir, 'categories.json'), 'r'))
        else:
            self._cls_vocab = None

        # Load metadata
        with open(metadata_file_path, encoding='utf-8') as f:
            self._metadata = _read_meta(f, self._hparams)

        hours = sum([int(x['l']) for x in self._metadata]) / hparams.sr / 3600
        logging.info('Loaded metadata from %s for %d examples (%.2f hours)' % (datadir, len(self._metadata), hours))

        if self._world_size > 1:
            self._metadata = self._metadata[self._rank::self._world_size]
            logging.info("%d samples after sharding" % len(self._metadata))

        self._metadata.sort(key=lambda x: x['n'])
        if shuffle:
            self.rand.shuffle(self._metadata)

        if source_lang:
            self._metadata = [x for x in self._metadata if x['sl'] in source_lang]
            logging.info("%d samples after filtering source language" % len(self._metadata))
        if target_lang:
            self._metadata = [x for x in self._metadata if x['tl'] in target_lang]
            logging.info("%d samples after filtering target language" % len(self._metadata))
        if filter_samples and hparams.input_length_final_upper_bound > 0:
            self._metadata = [x for x in self._metadata if
                              hparams.input_length_final_lower_bound <= int(x['l'])
                              <= self._hparams.input_length_final_upper_bound]
            logging.info("%d samples after filtering length" % len(self._metadata))

        if self._hparams.filter_by_charset:
            self._metadata = [x for x in self._metadata if x['tl'] in ['u', 'tag'] or
                              not any([c not in charset[x['tl']] for c in x['t']])]
            logging.info("%d samples after filtering charset" % len(self._metadata))

        if self._hparams.target_length_lower_bound > 0 and self._hparams.data_format not in ['nltLa', 'DlStLa']:
            self._metadata = [x for x in self._metadata if len(x['t']) > self._hparams.target_length_lower_bound]
            logging.info("%d samples after filtering target length" % len(self._metadata))

        if hparams.data_format == 'nltLa':
            self._metadata = [x for x in self._metadata if all(
                x['t'].split(',')[i] in self._cls_vocab[i] for i in range(len(self._cls_vocab)))]
            logging.info("%d samples after filtering categories" % len(self._metadata))

        self.lang_pairs = set([(x['sl'], x['tl']) for x in self._metadata])

    def __len__(self):
        return len(self._metadata)

    def state_dict(self):
        with self._lock:
            state = {'rand': self.rand.get_state(), 'offset': self._offset, 'epoch': self._epoch}

            if hasattr(self, '_adapt_offset'):
                state['adapt_offset'] = self._adapt_offset
            logging.info("Dumped feeder " + os.path.split(self._datadir)[-1] + " state: " + str(state['offset']))
            return state

    def load_state_dict(self, state):
        logging.info("Loaded feeder " + os.path.split(self._datadir)[-1] + " state: " + str(state['offset']))
        self.rand.set_state(state['rand'])
        self._offset = state['offset']
        self._epoch = state['epoch']
        if hasattr(self, '_adapt_offset'):
            state['adapt_offset'] = self._adapt_offset

    def get_examples(self, bucket_size):
        examples = []
        with self._lock:
            for i in range(bucket_size):
                r = self._get_next_example()
                if r is not None:
                    examples.append(r)
                if self.finished:
                    break
            if self._n_skip:
                logging.info("Skipped %d samples" % self._n_skip)
                self._n_skip = 0
        return examples

    def get_next_group(self, bucket_size):
        tic = time.time()
        examples = self.get_examples(bucket_size)
        examples.sort(key=lambda x: len(x['input']))
        batches = _pack_into_batches(examples, self._batch_size, self._batch_frame_limit, self._batch_quad_frame_limit)
        if self._shuffle:
            self.rand.shuffle(batches)

        for i, batch in enumerate(batches):
            batch = _prepare_batch(batch, tokenizer=self._tokenizer, extractor=self._extractor, hparams=self._hparams,
                                   decoder_tokenizer=self._decoder_tokenizer)
            batches[i] = batch
        logging.info("Packed %d batches with %d samples in %.2f sec from %s" %
                     (len(batches), len(examples), time.time() - tic, self._datadir))
        return batches
    
    def _enqueue_next_group(self):
        batches = self.get_next_group(self._hparams.bucket_size)
        for batch in batches:
            self.queue.put(dict([(name, self.proto[name](batch[name])) for name in self.proto if name in batch]))
        if self.finished and self._max_epoch > 0:
            self.queue.put(None)
            raise StopIteration

    def _get_next_example(self):
        while True:
            meta = self._metadata[self._offset]
            self._offset += 1
            if self._offset >= len(self._metadata):
                self._offset = 0
                self._epoch += 1
                if self._epoch >= self._max_epoch and self._max_epoch > 0:
                    self.finished = True
                if self._shuffle:
                    self.rand.shuffle(self._metadata)
            if self._filter_samples and self.skip_meta(meta):
                self._n_skip += 1
                continue
            try:
                return extract_meta(meta, self._datadir, self._hparams, self._cls_vocab)
            except:
                tb.print_exc()
            if self.finished:
                return None

    def skip_meta(self, meta):
        if self.global_step == -1 or self.global_step >= self._hparams.data_warmup_steps:
            return False
        if self._hparams.input_length_upper_bound > 0 and \
                not self._hparams.input_length_lower_bound <= int(meta['l']) <= self._hparams.input_length_upper_bound:
            return True
        return False

    # Methods for loading all data, not just the next batch; used for evaluation
    def _get_all_examples(self):
        examples = []
        while True:
            example = self._get_next_example()
            examples.append(example)
            if self._epoch == 1:
                self._epoch = 0
                break
        if self._n_skip:
            logging.info("Skipped %d samples" % self._n_skip)
            self._n_skip = 0
        return examples

    def get_all_batches(self, exclude=None):
        tic = time.time()
        logging.info("Loading all %d samples from %s" % (len(self._metadata), self._datadir))
        examples = self._get_all_examples()
        logging.info("Loaded %d examples from %s in %.2f sec" % (len(examples), self._datadir, time.time() - tic))
        examples = [x for x in examples if exclude is None or x['name'] not in exclude]
        examples.sort(key=lambda x: len(x['input']))
        # examples[0], examples[5] = examples[5], examples[0]
        # examples[2], examples[7] = examples[7], examples[2]

        all_batches = []
        for sl, tl in self.lang_pairs:
            sel_examples = [x for x in examples if x['src_lang'] == sl and x['tgt_lang'] == tl]
            batches = _pack_into_batches(sel_examples, self._batch_size, self._batch_frame_limit, self._batch_quad_frame_limit,
                                         self._single)
            all_batches.extend(batches)
        return all_batches

    def prepare_all_batches(self):
        tic = time.time()
        batches = self.get_all_batches()
        ret = []
        n_samples = 0
        for batch in batches:
            batch = _prepare_batch(batch, tokenizer=self._tokenizer, extractor=self._extractor, hparams=self._hparams,
                                   decoder_tokenizer=self._decoder_tokenizer)
            n_samples += len(batch['inputs'])
            ret.append(batch)
        self.data = ret
        logging.info("Prepared all %d batches of %d samples in %.2f sec from %s" %
                     (len(ret), n_samples, time.time() - tic, self._datadir))

    def fetch_data(self, exclude=None):
        if not hasattr(self, 'data'):
            self.prepare_all_batches()
        if exclude is None:
            if hasattr(self, 'eval_batches'):
                return self.eval_batches
            data = self.data
        else:
            data = self.prepare_all_batches(self.get_all_batches(exclude))
        # self._shuffle = False
        if self._shuffle:
            self.rand.shuffle(data)
        for batch in data:
            for name in batch:
                if name in self.proto:
                    batch[name] = self.proto[name](batch[name])
        self.eval_batches = data
        return data


class MultiFeeder(ProtoFeeder):
    def __init__(self, processor, hparams, feeders: List[Feeder], rank, world_size, shuffle=True,
                 sample_by_length=True, seed=2**32 - 1, group_ratio=None):
        super(MultiFeeder, self).__init__(processor=processor, hparams=hparams,
                                          rank=rank, world_size=world_size, seed=seed)
        self.feeder = feeders
        if sample_by_length:
            lengths = np.asarray([len(f) for f in feeders])
            self.weight = lengths / lengths.sum()
        elif group_ratio:
            self.weight = np.asarray(group_ratio) / sum(group_ratio)
        else:
            self.weight = np.ones(len(feeders)) / len(feeders)
        self.hp = hparams
        self.proto = get_input_proto(hparams)
        self._shuffle = shuffle
        self._sample_by_length = sample_by_length

    def get_next_group(self, bucket_size):
        choices = self.rand.choice(range(len(self.feeder)), size=bucket_size, p=self.weight)
        sources = [0 for _ in range(len(self.feeder))]
        for c in choices:
            sources[c] += 1
        batches = []
        for i, f in enumerate(self.feeder):
            f_batches = f.get_next_group(sources[i])
            batches.extend(f_batches)
        if self._shuffle:
            self.rand.shuffle(batches)
        return batches
    
    def _enqueue_next_group(self):
        tic = time.time()
        batches = self.get_next_group(self._hparams.bucket_size)
        logging.info("MultiFeeder: Packed %d batches in %.2f sec" %
                     (len(batches), time.time() - tic))
        for batch in batches:
            self.queue.put(dict([(name, self.proto[name](batch[name])) for name in self.proto if name in batch]))
    
    def get_all_batches(self):
        batches = []
        for f in self.feeder:
            batches.extend(f.get_all_batches())
        return batches

    def __len__(self):
        return sum([len(f) for f in self.feeder])

    def prepare_all_batches(self):
        batches = []
        for i, f in enumerate(self.feeder):
            if not hasattr(f, 'data'):
                f.prepare_all_batches()
            batches.extend(f.data)
        self.data = batches

    def fetch_data(self):
        if not hasattr(self, 'data'):
            self.prepare_all_batches()
        if hasattr(self, 'eval_batches'):
            return self.eval_batches
        data = self.data
        if self._shuffle:
            self.rand.shuffle(data)
        for batch in data:
            for name in batch:
                if name in self.proto:
                    batch[name] = self.proto[name](batch[name])
        self.eval_batches = data
        return data

    def state_dict(self):
        states = []
        for f in self.feeder:
            states.append(f.state_dict())
        return states

    def load_state_dict(self, state):
        for i, f in enumerate(self.feeder):
            f.load_state_dict(state[i])

    @property
    def global_step(self):
        return self._global_step

    @global_step.setter
    def global_step(self, value):
        self._global_step = value
        for f in self.feeder:
            f.global_step = value


def _read_meta(meta_file, hp):
    meta_list = []
    format = hp.data_format
    for line in meta_file:
        parts = line.strip().split('|')
        if len(parts) != len(format):
            parts = line.strip().split('\t')
        if format == 'nlt':
            name, length, text = parts
            item_dict = {'n': name, 'l': int(length), 't': text, 'type': 's2s'}
        elif format == 'Iltbas':
            id, length, text, tgt_lang, src_lang, source_text = parts
            name = id.split('.')[0]
            # src_lang = src_lang.split('-')[0]
            item_dict = {'n': name, 'l': int(length), 't': text, 'sl': src_lang, 'tl': tgt_lang, 's': source_text,
                         'id': id, 'type': 's2s'}
            if len(id.split('.')) == 4:
                l, r = int(id.split('.')[-2]), int(id.split('.')[-1])
                item_dict['start'] = l
                item_dict['end'] = r
        elif format == 'Il__as':
            id, length, _, _, src_lang, source_text = parts
            name = id.split('.')[0]
            # src_lang = src_lang.split('-')[0]
            item_dict = {'n': name, 'l': int(length), 't': source_text,
                         'sl': src_lang, 'tl': src_lang, 's': source_text, 'id': id, 'type': 's2s'}
            if len(id.split('.')) == 4:
                l, r = int(id.split('.')[-2]), int(id.split('.')[-1])
                item_dict['start'] = l
                item_dict['end'] = r
        elif format == 'nltLa':
            id, length, text, label, src_lang = parts
            name = id.split('.')[0]
            if hp.use_infergen and hp.infergen_mode == 'cls':
                label = label.split(',')
                label = ['%d_%s' % (i, l) for i, l in enumerate(label)]
                label = ','.join(label)
            # if '-' in src_lang:
            #     item_dict = {'n': name, 'l': int(length), 's': text, 't': label, 'sl': src_lang.split('-')[0],
            #                  'tl': 'u' + '_' + src_lang, 'type': 'cls'}
            # else:
            item_dict = {'n': name, 'l': int(length), 's': text, 't': label, 'sl': src_lang, 'tl': 'u',
                         'type': 'cls'}
            if len(id.split('.')) == 4:
                l, r = int(id.split('.')[-2]), int(id.split('.')[-1])
                item_dict['start'] = l
                item_dict['end'] = r
        elif format == 'DlStLa':
            # if len(parts) != 6:
            #     print()
            id, length, seg_length, text, label, src_lang = parts
            item_dict = {'n': id, 'l': int(length), 's': text, 't': label, 'sl': src_lang, 'tl': 'tag',
                         'S': int(seg_length), 'type': 'cls'}
        else:
            raise ValueError('Invalid format for _read_meta: %s' % format)
        meta_list.append(item_dict)
    return meta_list


def _pack_into_batches(examples, batch_size, batch_frame_limit, batch_quad_frame_limit, single=False):
    batches = [[]]
    for sample in examples:
        input_len = len(sample['input'])
        if single or len(batches[-1]) == batch_size or \
                (len(batches[-1]) + 1) * input_len > batch_frame_limit or \
                (len(batches[-1]) + 1) * input_len * input_len > batch_quad_frame_limit:
            batches.append([])
        batches[-1].append(sample)
    return batches


def _prepare_batch(batch, tokenizer: Wav2Vec2CTCTokenizer, extractor: Wav2Vec2FeatureExtractor,
                   hparams, decoder_tokenizer=None):
    input_lengths = np.asarray([len(x['input']) for x in batch], dtype=np.int32)
    results = {'input_lengths': input_lengths,
               'names': [x['name'] for x in batch],
               'src_lang': [language_id[x['src_lang']] for x in batch],
               'tgt_lang': [language_id[x['tgt_lang']] for x in batch]}
    if hparams.input_type == 'audio':
        inputs = extractor([x['input'] for x in batch], padding=True, pad_to_multiple_of=8, sampling_rate=hparams.sr)
        results['inputs'] = np.asarray(inputs.data['input_values'])
        if 'attention_mask' in inputs.data:
            results['input_masks'] = np.asarray(inputs.data['attention_mask'])
    else:
        max_length = max([len(x['input']) for x in batch])
        inputs = np.zeros((len(batch), max_length) + batch[0]['input'].shape[1:], dtype=np.float32)
        mask = np.zeros((len(batch), max_length), dtype=np.int32)
        for i, x in enumerate(batch):
            inputs[i, :len(x['input'])] = x['input']
            mask[i, :len(x['input'])] = 1
        results['inputs'] = inputs
        results['input_masks'] = mask

    if 'label' in batch[0]:
        if batch[0].get('type', '') == 's2s':
            labels = tokenizer([x['label'].upper() if hparams.upper_only else x['label'] for x in batch], padding=True)
            results['labels'] = np.asarray(labels.data['input_ids'])
            results['label_lengths'] = np.asarray([len(x['label']) for x in batch], dtype=np.int32)
            results['label_masks'] = np.asarray(labels.data['attention_mask'])

            results['labels'] = np.where(np.asarray(labels.data['attention_mask']) == tokenizer.pad_token_id,
                                         -100, results['labels'])

            if hparams.use_decoder:
                labels = decoder_tokenizer([x['label'] for x in batch], padding=True)
                results['decoder_labels'] = np.asarray(labels.data['input_ids'])
                results['decoder_label_masks'] = np.asarray(labels.data['attention_mask'])
                results['decoder_label_lengths'] = np.asarray([x.sum() for x in results['decoder_label_masks']],
                                                              dtype=np.int32)

                results['decoder_labels'] = np.where(np.asarray(labels.data['input_ids'])
                                                     == decoder_tokenizer.pad_token_id, -100, results['decoder_labels'])
        elif batch[0].get('type', '') == 'cls':
            # results['labels'] = np.asarray([x['label'] for x in batch])
            results['label_lengths'] = np.asarray([len(x['label']) for x in batch], dtype=np.int32)
            max_length = max(max(results['label_lengths']), 2) # Avoid empty tensor
            labels = np.zeros((len(batch), max_length), dtype=np.int32)
            for i, x in enumerate(batch):
                labels[i, :len(x['label'])] = x['label']
                labels[i, len(x['label']):] = -100
            results['labels'] = labels
    if 'dataset_name' in batch[0]:
        results['dataset_names'] = [x['dataset_name'] for x in batch]
    if 'input_segment' in batch[0]:
        results['input_segments'] = np.asarray([x['input_segment'] for x in batch])
    results['type'] = batch[0]['type']
    return results

zipfiles = {}
def get_audiofile_spec(datadir, sub_dir, name):
    if os.path.exists(os.path.join(datadir, sub_dir + '.zip')):
        if (datadir, sub_dir) not in zipfiles:
            zipfiles[(datadir, sub_dir)] = zipfile.ZipFile(os.path.join(datadir, sub_dir + '.zip'))
        file = zipfiles[(datadir, sub_dir)].open(name)
        return io.BytesIO(file.read())
    else:
        if os.path.exists(os.path.join(datadir, sub_dir, name)):
            return open(os.path.join(datadir, sub_dir, name), 'rb')
        else:
            sub_sub_dir = name[len(name.split('_')[0]) + 1:][:7]
            if os.path.exists(os.path.join(datadir, sub_dir, sub_sub_dir, name)):
                return open(os.path.join(datadir, sub_dir, sub_sub_dir, name), 'rb')
            else:
                if len(name.split('_')) == 3:
                    sub_sub_dir = name.split('_')[1]
                    return get_audiofile_spec(os.path.join(datadir, sub_dir), sub_sub_dir, name)
                else:
                    raise ValueError('File not found: %s, %s, %s' % (datadir, sub_dir, name))

def get_audiofile(datadir, sub_dir, name):
    for suffix in ['.flac', '.wav', '', '.pkl']:
        try:
            audio_file = get_audiofile_spec(datadir, sub_dir, name + suffix)
            return audio_file
        except:
            pass

def read_file(f, start=0, stop=None, input_type='audio'):
    if input_type == 'audio':
        return sd.read(f, start=start, stop=stop)[0]
    elif input_type == 'pickle':
        return pickle.load(f)
    else:
        raise ValueError('Unknown input type: %s' % input_type)


def extract_meta(meta, datadir, hparams, cls_vocab=None, cached=False):
    name = meta['n']
    results = {'name': meta.get('id', name), 'dataset_name': os.path.split(datadir)[-1]}

    if hparams.data_format == 'DlStLa':
        names = name.split(',')
        audio_file = [get_audiofile(datadir, n.split('_')[0], n) for n in names]

        input_data = [read_file(a, input_type=hparams.input_type) for a in audio_file]
        input_data = np.concatenate(input_data)
        results['input_segment'] = meta['S']
    else:
        audio_file = get_audiofile(datadir, name.split('_')[0], name)
        if 'start' not in meta:
            input_data = read_file(audio_file, input_type=hparams.input_type)
        else:
            input_data = read_file(audio_file, start=meta['start'], stop=meta['end'], input_type=hparams.input_type)
    results['input'] = input_data
    if hparams.data_format in ['nltLa']:
        results['label'] = [cls_vocab[i][x] for i, x in enumerate(meta['t'].split(','))]
        results['type'] = 'cls'
    elif hparams.data_format in ['DlStLa']:
        results['label'] = json.loads(meta['t']) # [N_spans * 2]
        if len(results['label']) == 0:
            results['label'] = [0, 0]
        results['type'] = 'cls'
    else:
        results['label'] = meta['t']
        results['type'] = 's2s'
    results['length'] = meta['l']
    if meta['l'] != len(results['input']):
        assert np.abs(meta['l'] / len(results['input']) - 1) < 0.05 # It may happen due to rounding errors, but not too large
        meta['l'] = len(results['input'])
    if 'sl' in meta:
        results['src_lang'] = meta['sl']
        results['tgt_lang'] = meta['tl']
        results['source_text'] = meta['s']
    return results


def get_input_proto(config):
    keys = {'inputs': torch.FloatTensor, 'input_lengths': torch.LongTensor,
            'labels': torch.LongTensor, 'label_lengths': torch.LongTensor,
            'src_lang': torch.LongTensor, 'tgt_lang': torch.LongTensor,
            'names': list, 'dataset_names': list, 'type': str}
    if config.use_attention_mask:
        keys['input_masks'] = torch.LongTensor
        keys['label_masks'] = torch.LongTensor
    if config.use_decoder:
        keys['decoder_labels'] = torch.LongTensor
        keys['decoder_label_lengths'] = torch.LongTensor
        keys['decoder_label_masks'] = torch.LongTensor
    if config.data_format == 'DlStLa':
        keys['input_segments'] = torch.LongTensor
    return keys

def parse_meta_list(meta, data_dir, default_file):
    train_meta = [os.path.join(t, default_file) for t in data_dir]
    if meta:
        t_meta = meta.split(':')
        for i, t in enumerate(t_meta):
            if t_meta[i] == '.':
                train_meta[i] = None
            elif t_meta[i] != '':
                if os.path.exists(t_meta[i]):
                    train_meta[i] = t_meta[i]
                elif os.path.exists(os.path.join(data_dir[i], t_meta[i])):
                    train_meta[i] = os.path.join(data_dir[i], t_meta[i])
                else:
                    raise ValueError('Meta file not found: %s' % t_meta[i])
    return train_meta


def get_feeder(args, hp, rank, world_size, get_train=True, get_eval=True, shuffle_eval=True, reduce_eval_batch=True):
    import models.build as build
    src_lang = args.src_lang.split(':') if args.src_lang else None
    tgt_lang = args.tgt_lang.split(':') if args.tgt_lang else None

    if args.datasets:
        datasets = args.datasets.split(':')
        data_dir = [os.path.join(args.data_dir, d) for d in datasets]
    else:
        data_dir = [args.data_dir]

    vocab_path = args.vocab_path if args.vocab_path else os.path.join(args.data_dir, 'vocab.json')
    hparams = [copy.copy(hp) for _ in data_dir]
    if ':' in hp.data_format:
        data_format = hp.data_format.split(':')
        assert len(data_format) == len(data_dir)
        for i, d in enumerate(data_format):
            hparams[i].data_format = d
    if not os.path.exists(vocab_path):
        vocab_path = os.path.join(data_dir[0], 'vocab.json')
    processor = build.build_processor(hp, vocab_path)

    if get_train:
        train_meta = parse_meta_list(args.train_meta, data_dir, 'meta.train.txt')
        train_feeders = []
        for i, d in enumerate(data_dir):
            if train_meta[i] is None:
                logging.info('No training data for %s' % d)
                continue
            feeder = Feeder(d, processor, train_meta[i], hparams=hparams[i],
                            rank=rank, world_size=world_size, shuffle=hp.shuffle_training_data,
                            source_lang=src_lang, target_lang=tgt_lang, seed=rank * len(data_dir) + i,
                            max_epoch=args.max_epoch)
            train_feeders.append(feeder)
        if len(train_feeders) == 1:
            feeder = train_feeders[0]
        elif hp.data_groups:
            groups = hp.data_groups.split(':')
            group_feeders = []
            k = 0
            for i, group_size in enumerate(groups):
                group_size = int(group_size)
                group_feeder = MultiFeeder(processor, hp, train_feeders[k: k + group_size], rank, world_size,
                                 shuffle=hp.shuffle_training_data, seed=2 ** 32 - (i + 1) * (rank + 1))
                group_feeders.append(group_feeder)
                k += group_size
            feeder = MultiFeeder(processor, hp, group_feeders, rank, world_size, shuffle=hp.shuffle_training_data,
                                 sample_by_length=False, seed=2 ** 32 - len(groups) * (rank + 1) - 1,
                                 group_ratio=[float(t) for t in hp.data_group_ratio.split(':')])
        else:
            feeder = MultiFeeder(processor, hp, train_feeders, rank, world_size,
                                 shuffle=hp.shuffle_training_data, seed=2 ** 32 - (rank + 1))
    else:
        feeder = None
    if get_eval:
        eval_meta = parse_meta_list(args.eval_meta, data_dir, 'meta.dev.txt')
        eval_feeders = []
        for i, d in enumerate(data_dir):
            if eval_meta[i] is None:
                logging.info('No eval data for %s' % d)
                continue
            feeder_eval = Feeder(d, processor, eval_meta[i], hparams=hparams[i], filter_samples=hp.eval_filter_samples,
                                 source_lang=src_lang, target_lang=tgt_lang,
                                 shuffle=shuffle_eval, seed=rank * len(data_dir) + i)
            if reduce_eval_batch:
                feeder_eval._batch_frame_limit /= 2  # Try to avoid OOM in eval
                feeder_eval._batch_quad_frame_limit /= 2
            eval_feeders.append(feeder_eval)
        if len(eval_feeders) == 1:
            feeder_eval = eval_feeders[0]
        else:
            feeder_eval = MultiFeeder(processor, hp, eval_feeders, 0, world_size, shuffle=shuffle_eval)
        # feeder_eval.get_all_batches()
    else:
        feeder_eval = None
    return feeder, feeder_eval, processor

def pick_eval_samples(batches, hp):
    from collections import defaultdict, Counter
    buckets = defaultdict(list)
    sample_cnt = Counter()
    for batch in batches:
        key = (batch['dataset_names'][0], batch['src_lang'][0].item(), batch['tgt_lang'][0].item())
        if sample_cnt[key] <= hp.max_eval_samples or batch['dataset_names'][0] in ['slurp_full', 'massive', 'nmsqa', 'minds14', 'gigawords']:
            buckets[key].append(batch)
            sample_cnt[key] += len(batch['names'])
    logging.info("Picked eval samples: " + '; '.join(["%s: %d" % (k, sample_cnt[k]) for k, v in buckets.items()]))
    results = []
    for v in buckets.values():
        results.extend(v)
    return results