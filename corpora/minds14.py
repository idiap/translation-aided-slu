#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import os, json, tqdm
import librosa, soundfile
from collections import Counter

in_dir = r"/local/scratch/3690/data/minds14/"
output_path = "/temp/3690/data/minds14"
sr = 16000

os.makedirs(output_path, exist_ok=True)

def convert(filename, lang, id):
    wavform = librosa.load(os.path.join(in_dir, 'audio', filename), sr=sr)[0]
    wavform = librosa.effects.trim(wavform, top_db=35)[0]
    os.makedirs(os.path.join(output_path, lang), exist_ok=True)
    soundfile.write(os.path.join(output_path, lang, id + '.flac'), wavform, sr)
    return len(wavform)

def prepare_data(langs):
    files = {}
    for split in ['train', 'dev', 'test']:
        files[split] = open(os.path.join(output_path, "meta.%s.txt" % split), "w", encoding='utf-8')
    audio_files = {}
    audio_meta = {}
    lang_cnt = Counter()
    for split in ['train', 'dev', 'test']:
        for lang in langs:
            meta = open(os.path.join(in_dir, 'split', '%s_%s.tsv' % (split, lang))).read().splitlines()
            meta = [line.split('\t') for line in meta]
            for file in meta:
                if file[0] not in audio_files:
                    lang_cnt[lang] += 1
                    audio_files[file[0]] = "%s_%010d" % (lang, lang_cnt[lang])
                    audio_meta[file[0]] = (split, lang, file[0], file[1], file[2])
                else:
                    print("Found duplicate file %s" % file['file'])
                    continue

    audio_dur = {}
    for filename, id in tqdm.tqdm(audio_files.items()):
        audio_dur[id] = convert(filename, audio_meta[filename][1], id)
    print("Total %d files, %.2f H" % (len(audio_files), sum(audio_dur.values()) / sr / 3600))

    for filename, id in tqdm.tqdm(audio_files.items()):
        split, lang, filename, text, intent = audio_meta[filename]
        info = '|'.join([id, str(audio_dur[id]), text, intent, lang])
        files[split].write(info + '\n')
    intents = list(set([audio_meta[filename][4] for filename in audio_files]))
    intents.sort()
    intents = dict([(intent, i) for i, intent in enumerate(intents)])
    json.dump(intents, open(os.path.join(output_path, "categories.json"), "w"))

if __name__ == '__main__':
    prepare_data(['en-US', 'en-GB', 'en-AU', 'fr-FR'])