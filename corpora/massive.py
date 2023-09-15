#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import glob
import os
import json

import soundfile
import tqdm
import soundfile as sd
import librosa
from collections import Counter, defaultdict
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from matplotlib import pyplot as plt
from synthesize import synthesize_speech, voices
import numpy as np

sr = 16000

# Put the fr-FR.jsonl file to the same directory as the SLURP dataset
meta_dir = r"/home/3690/repo/slurp/dataset/slurp/"
tmp_dir = r"/local/scratch/3690/data/slurp_fr"
out_dir = r"/temp/3690/data/slurp_fr"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

def synthesize_audios(lang):
    meta = open(os.path.join(meta_dir, "%s.jsonl" % (lang))).read().splitlines()
    meta = [json.loads(line) for line in meta]

    ids = set()
    random.seed(0)
    used_voices = []
    executor = ProcessPoolExecutor(max_workers=8)
    futures = []
    for line in tqdm.tqdm(meta):
        voice = random.choice(voices[lang])
        used_voices.append(voice)
        out_path = os.path.join(tmp_dir, "%s_%s.flac" % (lang, line['id']))
        # print(line['utt'])
        if os.path.exists(out_path):
            continue
        if line['id'] in ids:
            print("Duplicate id: %s" % line['id'])
        futures.append(
            executor.submit(
                partial(synthesize_speech, line['utt'], lang, out_path, voice)))
    [f.result() for f in tqdm.tqdm(futures)]
    json.dump(used_voices, open(os.path.join(tmp_dir, "used_voices.json"), "w"))

def convert(from_path, to_path, sr):
    audio, orig_sr = sd.read(from_path)
    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    sd.write(to_path, audio, sr)

def prepare_data(lang):
    meta = open(os.path.join(meta_dir, "%s.jsonl" % (lang))).read().splitlines()
    meta = [json.loads(line) for line in meta]
    prefix = lang.split('-')[0]


    executor = ProcessPoolExecutor(max_workers=8)
    futures = []
    os.makedirs(os.path.join(out_dir, prefix), exist_ok=True)

    audio_files = {}
    audio_dur = {}
    split_cnt = defaultdict(int)

    for m in tqdm.tqdm(meta):
        assert m['id'] not in audio_files
        audio_files[m['id']] = "%s_%010d.flac" % (prefix, len(audio_files))
        futures.append(executor.submit(partial(convert,
                                               os.path.join(tmp_dir, "%s_%s.flac" % (lang, m['id'])),
                                               os.path.join(out_dir, prefix, audio_files[m['id']]), sr)))

    [f.result() for f in tqdm.tqdm(futures)]


    files = {}
    for split in ['train', 'dev', 'test']:
        files[split] = open(os.path.join(out_dir, "meta.%s.txt" % split), "w", encoding='utf-8')

    for m in tqdm.tqdm(meta):
        split = m['partition']
        id = audio_files[m['id']]
        dur = sd.info(os.path.join(out_dir, prefix, id)).frames
        audio_dur[id] = dur
        intent = (m['scenario'], m['intent'][len(m['scenario']) + 1:])
        split_cnt[split] += 1
        files[split].write('|'.join([id, str(dur), m['utt'], ','.join(intent), prefix]) + '\n')

    json.dump(audio_files, open(os.path.join(out_dir, "audio_files.json"), "w"))

    plt.hist([t / sr for t in audio_dur.values()], bins=100)
    plt.title(lang)
    plt.show()

    print("Total %d utterances of %.2fH" % (len(meta), sum(audio_dur.values()) / sr / 3600))
    print("Split counts: %s" % split_cnt)

def downsample(out_name, max_samples=10):
    import random
    meta = open(os.path.join(out_dir, "meta.train.txt"), "r", encoding='utf-8').read().splitlines()

    intents = defaultdict(list)
    for m in meta:
        intent = tuple(m.split('|')[3].split(','))
        intents[intent].append(m)

    for intent in intents:
        random.seed(0)
        random.shuffle(intents[intent])
        intents[intent] = intents[intent][:max_samples]

    data = []
    for intent in intents:
        data += intents[intent]
    data.sort()
    print("Downsampled from %d to %d utterances with max_samples %d" % (len(meta), len(data), max_samples))
    fw = open(os.path.join(out_dir, out_name), "w", encoding='utf-8')
    for line in data:
        fw.write(line + '\n')


if __name__ == '__main__':
    pass
    synthesize_audios('fr-FR')
    prepare_data('fr-FR')
    downsample('meta.train.100s.txt', max_samples=100)
