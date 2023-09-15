#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import os, glob, shutil, json
import soundfile as sd
import librosa
from pydub import AudioSegment
import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import warnings
import re
import matplotlib.pyplot as plt
import tarfile
from zipfile import ZipFile
import pydub
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

db_path = "/temp/3690/covost2"
output_path = "/temp/3690/data/covost2"
os.makedirs(output_path, exist_ok=True)
sr = 16000

tmp_path = "/local/scratch/3690/tmp/covost2/"
os.makedirs(tmp_path, exist_ok=True)
os.makedirs(os.path.join(tmp_path, 'convert'), exist_ok=True)

lang_pairs = {'fr': ['en']}

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def convert(from_path, to_path, sr):
    if os.path.exists(to_path):
        return sd.info(to_path).frames
    try:
        with warnings.catch_warnings():  # Suppress librosa warnings reading m4a
            warnings.simplefilter("ignore")
            sr_, audio = read(from_path, True)
            audio = librosa.resample(audio, orig_sr=sr_, target_sr=sr)
        sd.write(to_path, audio, sr)
        return len(audio)
    except:
        print("Fail to read file {}".format(from_path))
        return -1

import unicodedata
import re
def normalize(text):
    if '‘' in text:
        while True: # Ignore nested cases; hopefully there won't be one
            s = text.find('‘')
            if s == -1:
                break
            t = text.find('’', s)
            if t != -1:
                if s <= text.find('‘', s+1) < t:
                    print("Nested case: {}".format(text))
                text = text[:s] + text[s+1:t] + text[t+1:]
            else:
                break
    text = text.replace("`s ", "'s ").replace("[Ms.]", "")
    map = {"Ō": "O", "ō": "o", "Ū": "U", "ū": "u", "∞": " infinity ",
           "&": " and ", "°": "º", '′': "'", '’': "'"}
    for ch in map:
        text = text.replace(ch, map[ch])
    for i, ch in enumerate(text):
        if ch in [chr(0x0300), chr(0x0301)] and (i == 0 or text[i - 1] == ' ') and text[i+1].isalpha() and i+1 < len(text):
            text = text[:i] + text[i+1] + text[i] + text[i+2:]
    text = unicodedata.normalize('NFC', text)
    for ch in "\"|«»‹›“”()_…—–/=\u200b^{}<>±§☉~∅[]@\u0320„*†×‘":
        text = text.replace(ch, ' ')
    text = re.sub(r'\s+', ' ', text).strip(chr(0x300)).strip('"').strip("'").strip()

    return text


chs = 'ABCDEFGHIJKLMNOPQRSTUVWYXZabcdefghijklmnopqrstuvwxyz0123456789!\',-.:;? "' \
      + 'áàâæçéèêëîïôœùûüÿŸÜÛÙŒÔÏÎËÊÈÉÇÆÂÀóҪäöść´`' + "%·$€£"

def prepare_data():
    files = {}
    for split in ['train', 'dev', 'test']:
        files[split] = open(os.path.join(output_path, "meta.%s.txt" % split), "w", encoding='utf-8')
    vocab = set()
    n_total_samples = 0
    n_total_sec = 0
    futures = []
    executor = ProcessPoolExecutor(max_workers=4)
    for source_lang in lang_pairs:
        audio_files = {}
        os.makedirs(os.path.join(output_path, source_lang), exist_ok=True)

        for target_lang in lang_pairs[source_lang]:
            for split in ["train", "dev", "test"]:
                segments = open(os.path.join(db_path, source_lang, 'covost_v2.%s_%s.%s.tsv' % (source_lang, target_lang, split))).read().splitlines()
                for segment in tqdm.tqdm(segments[1:]):
                    segment = segment.strip().split('\t')
                    file = segment[0]
                    if file not in audio_files:
                        audio_files[file] = "%s_%010d" % (source_lang, len(audio_files))
                    else:
                        print("Found duplicate file %s" % file)
        tar = tarfile.open(os.path.join(db_path, source_lang, source_lang + '.tar.gz'), mode='r|*')
        bar = tqdm.tqdm(total=len(audio_files))
        extracted = set()
        for f in tar:
            if f.name.startswith('clips/') and f.name[len('clips/'):] in audio_files:
                file = f.name[len('clips/'):]
                tar.extract(f, path=tmp_path)
                bar.update(1)
                extracted.add(file)
        if len(extracted) != len(audio_files):
            print("Missing files: %s" % (set(audio_files) - extracted))
        bar.close()
        for file in tqdm.tqdm(audio_files):
            # futures.append(convert(os.path.join(tmp_path, 'clips', file),
            #                 os.path.join(tmp_path, 'convert', audio_files[file] + '.flac'), sr))
            futures.append(
                executor.submit(
                    partial(convert, os.path.join(tmp_path, 'clips', file),
                            os.path.join(tmp_path, 'convert', audio_files[file] + '.flac'), sr)))
        # audio_dur = [f for f in futures]
        audio_dur = [f.result() for f in tqdm.tqdm(futures)]
        audio_dur = dict([(k, l) for k, l in zip(audio_files, audio_dur) if l != -1])


        zipfile = ZipFile(os.path.join(output_path, source_lang + '.zip'), mode='w')
        for file in tqdm.tqdm(audio_dur):
            zipfile.write(os.path.join(tmp_path, 'convert', audio_files[file] + '.flac'),
                          arcname=audio_files[file] + '.flac')

        sec_cnt = 0
        n_cnt = 0
        for target_lang in lang_pairs[source_lang]:
            for split in ["train", "dev", "test"]:
                segments = open(os.path.join(db_path, source_lang, 'covost_v2.%s_%s.%s.tsv' % (source_lang, target_lang, split))).read().splitlines()
                for segment in tqdm.tqdm(segments[1:]):
                    segment = segment.strip().split('\t')
                    file = segment[0]
                    if file not in audio_dur:
                        continue

                    if 'TO ' in segment[2] and ' REMOVE' in segment[2]:
                        continue

                    id = audio_files[file] + '.' + target_lang
                    source_text = normalize(segment[1])
                    target_text = normalize(segment[2])

                    n_cnt += 1
                    sec_cnt += audio_dur[file] / sr

                    info = '|'.join([id, str(audio_dur[file]), target_text, target_lang, source_lang, source_text])
                    files[split].write(info + '\n')

                    vocab = vocab.union(set(source_text)).union(set(target_text))

        print("%s: Total %d files (%d skip), %.2f H" % (source_lang, n_cnt, len(audio_files) - n_cnt, sec_cnt / 3600))
        plt.hist([t / sr for t in audio_dur.values()], bins=100)
        plt.title(source_lang)
        plt.show()

        n_total_samples += len(audio_files)
        n_total_sec += sec_cnt

        json.dump(audio_files, open(os.path.join(output_path, source_lang + '.audio_files.json'), 'w'), indent=1)

    print("Total %d paired samples, %.2f hours" % (n_total_samples, n_total_sec / 3600))
    vocab = ['[PAD]'] + sorted(list(vocab)) + ["'", '|', '[UNK]']
    vocab = dict([(c, i) for i, c in enumerate(vocab)])
    open(os.path.join(output_path, 'vocab.json'), 'w').write(json.dumps(vocab, indent=1, ensure_ascii=False))

if __name__ == '__main__':
    prepare_data()
