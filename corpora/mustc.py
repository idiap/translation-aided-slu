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
import yaml

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

db_path = "/local/scratch/3690/data/mustc/"
tmp_path = "/local/scratch/3690/tmp/mustc/"
output_path = "/temp/3690/data/mustc"
os.makedirs(output_path, exist_ok=True)
os.makedirs(tmp_path, exist_ok=True)
sr = 16000

lang_pairs = {'en': ['fr'], 'fr': ['en']}
modes = {('en', 'fr'): 'mustc', ('fr', 'en'): 'mtedx'}

def convert(from_path, to_path, sr):
    with warnings.catch_warnings():  # Suppress librosa warnings reading m4a
        warnings.simplefilter("ignore")
        audio, orig_sr = sd.read(from_path)
        if audio.shape[-1] == 2:
            audio = librosa.to_mono(audio.T)
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    sd.write(to_path, audio, sr)

import unicodedata
import re
def normalize(text):
    map = {"Ō": "O", "ō": "o", "Ū": "U", "ū": "u", "∞": " infinity ",
           "&": " and ", '′': "'", '’': "'", "\u00ad": '-', 'о': 'o', 'ĂŠ': 'é',
           'ᵉ': "º", " ´ ": "'", "˚": "°"}
    for ch in map:
        text = text.replace(ch, map[ch])
    text = unicodedata.normalize('NFC', text)
    for skip in ['Swahili)', '(Vidéo', 'Gujarati', '(In Dutch)', '(Khmer)', '(Kru English)', 'อ', '送', 'प', '\u05d8']:
        if skip in text:
            return ''
    if re.match(r"\(.*\)", text.strip()) or re.match(r"\[.*\]", text.strip()):
        return ''
    for ch in "♫♪«»‹›“‘”–—…\"«»[]„_¿®¡*•ʾ©\u0080\u0081™\ufffd\u007f|\u0092̐\u0090\t\u0094\u200b":
        text = text.replace(ch, '')
    for sub in ['<i>', '</i>', 'VoilĂ !']:
        text = text.replace(sub, '')
    for ch in "Žˆ‰ž":
        text = text.replace(ch, '')
    text = re.sub(r'\s+', ' ', text).strip(chr(0x300)).strip('"').strip("'").strip()
    return text


chs = 'ABCDEFGHIJKLMNOPQRSTUVWYXZabcdefghijklmnopqrstuvwxyz0123456789!\',-.:;?/’ º²()=+@' \
      + "%·$€£#¥" + 'àâæçéèêëîïôœùûüÿŸÜÛÙŒÔÏÎËÊÈÉÇÆÂÀ' \
      #+ 'āáàâæçéèêëîíïôœùûüÿŸÜÛÙŒÔÏÎËÊÈÉÇÆÁÂÀÖóҪäöúòšść´`ãñ'

def prepare_data():
    files = {}
    for split in ['train', 'dev', 'test']:
        files[split] = open(os.path.join(output_path, "meta.%s.txt" % split), "w", encoding='utf-8')
    vocab = set()
    n_total_samples = 0
    n_total_sec = 0
    futures = []
    executor = ProcessPoolExecutor(max_workers=8)
    for source_lang in lang_pairs:
        audio_files = {}
        audio_dur = []
        for target_lang in lang_pairs[source_lang]:
            if modes[(source_lang, target_lang)] == 'mustc':
                split_map = {'train': 'train', 'dev': 'dev', 'test': 'tst-COMMON'}
            elif modes[(source_lang, target_lang)] == 'mtedx':
                split_map = {'train': 'train', 'dev': 'valid', 'test': 'test'}
            for split in ["train", "dev", "test"]:
                split_path = os.path.join(db_path, source_lang + '-' + target_lang, 'data', split_map[split])
                segments = open(os.path.join(split_path, 'txt', split_map[split] + '.yaml')).read().splitlines()
                segments = [yaml.load(l, Loader=yaml.CLoader)[0] for l in segments]

                for i in range(len(segments)):
                    file = segments[i]['wav']
                    if modes[(source_lang, target_lang)] == 'mtedx':
                        file = file[:-4] + '.flac'
                    if segments[i]['duration'] > 20:
                        print(split, segments[i])
                    if file not in audio_files:
                        audio_files[file] = "%s_%010d" % (source_lang, len(audio_files))
                        # convert(os.path.join(split_path, 'wav', file),
                        #         os.path.join(tmp_path, audio_files[file] + '.flac'), sr)
                        futures.append(
                            executor.submit(
                                partial(convert, os.path.join(split_path, 'wav', file),
                                        os.path.join(tmp_path, audio_files[file] + '.flac'), sr)))

        [f.result() for f in tqdm.tqdm(futures)]

        sec_cnt = 0
        n_cnt = 0

        for target_lang in lang_pairs[source_lang]:
            for split in ["train", "dev", "test"]:
                split_path = os.path.join(db_path, source_lang + '-' + target_lang, 'data', split_map[split])
                segments = open(os.path.join(split_path, 'txt', split_map[split] + '.yaml')).read().splitlines()
                segments = [yaml.load(l, Loader=yaml.CLoader)[0] for l in segments]
                source_texts = open(os.path.join(split_path, 'txt',
                                              split_map[split] + '.' + source_lang)).read().splitlines()
                target_texts = open(os.path.join(split_path, 'txt',
                                            split_map[split] + '.' + target_lang)).read().splitlines()

                for i in range(len(segments)):
                    source_text = source_texts[i]
                    target_text = target_texts[i]

                    source_text = normalize(source_text)
                    target_text = normalize(target_text)

                    if not target_text or not source_text:
                        continue

                    for ch in source_text + target_text:
                        if ch not in chs:
                            print(ch, hex(ord(ch)), '|', source_text, '|', target_text)
                            break

                    file = segments[i]['wav']
                    if modes[(source_lang, target_lang)] == 'mtedx':
                        file = file[:-4] + '.flac'
                    start = int(segments[i]['offset'] * sr)
                    end = int((segments[i]['offset'] + segments[i]['duration']) * sr)

                    id = audio_files[file] + '.' + target_lang + '.' + str(start) + '.' + str(end)
                    audio_dur.append(segments[i]['duration'])

                    n_cnt += 1
                    sec_cnt += segments[i]['duration']

                    info = '|'.join([id, str(end - start), target_text, target_lang, source_lang, source_text])

                    files[split].write(info + '\n')

                    vocab = vocab.union(set(source_text)).union(set(target_text))

        print("%s: Total %d segments from %d files, %.2f H" % (source_lang, n_cnt, len(audio_files), sec_cnt / 3600))
        print("%d (%.2f min) samples too long" %
              (len([d for d in audio_dur if d > 20]), np.sum([d for d in audio_dur if d > 20]) / 60))
        plt.hist([t for t in audio_dur if t <= 20], bins=100)
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
