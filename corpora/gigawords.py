#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

from collections import defaultdict, Counter

import numpy as np
import tqdm
import json, os
import random
from synthesize import synthesize_speech, voices
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import re
import soundfile as sd

tmp_dir = r"/local/scratch/3690/data/gigaword"
out_dir = r"/temp/3690/data/gigaword"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)
sr = 16000

def normalize(text):
    text = text.replace('-lrb-', '(').replace('-rrb-', ')').replace('-lsb-', '[').replace('-rsb-', ']')\
        .replace('-lcb-', '{').replace('-rcb-', '}').replace("&amp;", "&").replace('{ ndash }', '--').\
        replace("/ {", "").replace("/ }", "").replace('{ rsquo }', "'").replace('{ lsquo }', "'")\
        .replace('{ ldquo }', '"').replace('{ rdquo }', '"').replace('{ emdash }', '--').replace('–', '--')
    text = (' ' + text + ' ').replace(" ' s ", "'s ").replace(" ' t ", "'t ").replace(" ' m ", "'m ").\
        replace(" ' ve ", "'ve ").replace(" ' re ", "'re ").replace(" ' d ", "'d ").replace(" ' ll ", "'ll ")
    text = text.replace(" n't ", "n't ").replace(" 's", "'s").replace(" 'd ", "'d ").\
        replace(" 'll ", "'ll ").replace(" 'm ", "'m ").replace(" 've ", "'ve ").replace(" 're ", "'re ")
    text = text.replace("\\", "").replace('_', ' ').replace('``', "''").replace('`', "'").\
        replace('~', '').replace('^', '')
    for ch in ",.!?:;)":
        text = text.replace(' ' + ch, ch)
    text = text.replace("( ", "(")
    text = text.replace("port-au - prince", "port-au-prince")
    text = (' ' + text + ' ').replace(' skorea', ' south korea').replace(' nkorea', ' north korea').\
        replace(' aa ', ' AA ').replace(' aaa ', ' AAA ')\
       .replace('u.s.', 'US').replace('u.s', 'US').replace('u.n.', 'UN').replace('u.n', 'UN').replace('u.k.', 'UK').replace('u.k', 'UK').replace('at&t', 'AT&T')
    for word in ['un', 'us', 'eu', 'au', 'roc']: # Fortunately in the dataset "us" is usually "US"
        text = re.sub(r'\b%s\b' % word, word.upper(), text)
    text = text.replace("í", "i")
    text = text.replace("+ postponed indefinitely +", "postponed indefinitely").\
        replace("+ speculative +", "speculative").replace("+ the people +", "the people").\
        replace("+ iran +" , "iran").replace("+ the star +", "the star") # Some manual fixes
    text = text.replace("port-AU-prince", "port-au-prince")
    while '---' in text:
        text = text.replace('---', '--')
    text = re.sub(r'\s+', ' ', text)
    return text.strip().strip('*').strip()

def prepare_texts():
    import datasets
    gigawords = datasets.load_dataset('gigaword')
    from matplotlib import pyplot as plt

    vocab = Counter()
    lengths = []
    sizes = {'train': 50000, 'validation': 1000, 'test': 1000}
    for split in ['validation', 'test', 'train']:
        docs = defaultdict(set)
        n_dup = n_skip = 0
        random.seed(0)
        ds = gigawords[split].shuffle(seed=0)
        for i, line in enumerate(tqdm.tqdm(ds)):
            line['document'] = normalize(line['document'])
            line['summary'] = normalize(line['summary'])
            to_skip = False
            for ch in ['#', 'UNK', '%', '{', '}', '*', '=', '@', '$']:
                if ch in line['document'] or ch in line['summary']:
                    to_skip = True
                    break
            if to_skip:
                n_skip += 1
                continue
            words = line['document'].split(' ')

            for text in [line['document'], line['summary']]:
                for ch in text:
                    if ch not in vocab or ch in ['*', '{', '_', '}', '%', '=', '^', '@', '–', '$']:
                        vocab[ch] += 1
                        print(ch, text)
            if any([w[0] == '-' and w[-1] == '-' and len(w) > 2 for w in words]):
                print(line['document'])
                n_skip += 1
                continue
            if len(words) > 30 or len(words) < 10:
                n_skip += 1
                continue
            if line['document'] in docs:
                n_dup += 1
                continue

            lengths.append(len(line['document']))
            docs[line['document']].add(line['summary'])
            vocab.update(list(line['document']))
            if len(docs) >= sizes[split]:
                break
        print(split, len(docs), "duplicates:", n_dup, "skipped:", n_skip, "pairs:", sum([len(v) for v in docs.values()]))
        fw = open(os.path.join(tmp_dir, "%s.jsonl" % split), "w")
        for doc, summary in docs.items():
            fw.write(json.dumps({'doc': doc, 'summary': list(summary)}) + '\n')
    print(len(lengths), sum(lengths), sum(lengths) / len(lengths))
    plt.hist(lengths)
    plt.show()

def convert(from_path, to_path, sr):
    import librosa
    audio, orig_sr = sd.read(from_path)
    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    sd.write(to_path, audio, sr)

def build_audio():
    executor = ProcessPoolExecutor(max_workers=16)
    futures = []

    audio_files = []
    used_voices = {'train': [], 'dev': [], 'test': []}

    random.seed(0)

    os.makedirs(os.path.join(tmp_dir, 'en'), exist_ok=True)
    split_map = {'train': 'train', 'dev': 'validation', 'test': 'test'}
    n_char = 0
    for split in ['dev', 'test', 'train']:
        meta = open(os.path.join(tmp_dir, "%s.jsonl" % split_map[split]), "r").readlines()
        for m in tqdm.tqdm(meta):
            m = json.loads(m)
            text = m['doc']
            voice = random.choice(voices['en-US'])
            used_voices[split].append(voice)
            name = "en_%010d" % (len(audio_files))
            sub_dir = name[len(name.split('_')[0]) + 1:][:7]
            os.makedirs(os.path.join(tmp_dir, 'en', sub_dir), exist_ok=True)
            out_path = os.path.join(tmp_dir, 'en', sub_dir, name + '.flac')
            audio_files.append((name, split, out_path, m['doc'], m['summary'][0]))
            n_char += len(text)
            if not os.path.exists(out_path):
                print(m['doc'])
                futures.append(
                    executor.submit(
                        partial(synthesize_speech, text, 'en-US', out_path, voice)))

    [f.result() for f in tqdm.tqdm(futures)]
    json.dump(used_voices, open(os.path.join(tmp_dir, "used_voices.json"), "w"))

    files = {}
    for split in ['train', 'dev', 'test']:
        files[split] = open(os.path.join(out_dir, "meta.%s.txt" % split), "w", encoding='utf-8')

    futures = []
    for name, split, path, doc, summary in tqdm.tqdm(audio_files):
        filename = os.path.split(path)[1]
        assert name + '.flac' == filename
        sub_dir = name[len(name.split('_')[0]) + 1:][:7]
        os.makedirs(os.path.join(out_dir, 'en', sub_dir), exist_ok=True)
        out_path = os.path.join(out_dir, 'en', sub_dir, name + '.flac')
        if not os.path.exists(out_path):
            # convert(path, out_path, sr)
            futures.append(executor.submit(partial(convert, path, out_path , sr)))

    [f.result() for f in tqdm.tqdm(futures)]

    for name, split, path, doc, summary in tqdm.tqdm(audio_files):
        dur = sd.info(out_dir + path[len(tmp_dir):]).frames
        files[split].write("|".join([name + '.en-sum', str(dur), summary.lower(), 'en-sum', 'en', doc.lower()]) + '\n')

if __name__ == '__main__':
    pass
    prepare_texts()
    build_audio()