#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import os, json
import datasets
import numpy as np
import soundfile as sd
import librosa
from matplotlib import pyplot as plt
import tqdm
import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import warnings
from collections import defaultdict
import shutil
import zipfile
import pandas as pd
import pickle
import math

tmp_dir = r"/local/scratch/3690/data/nmsqa"
out_dir = r"/temp/3690/data/nmsqa"
sr = 16000
os.makedirs(out_dir, exist_ok=True)

def convert(from_path, to_path, sr):
    with warnings.catch_warnings():  # Suppress librosa warnings reading mp3
        warnings.simplefilter("ignore")
        audio, _ = librosa.load(from_path, sr=sr)
    sd.write(to_path, audio, sr)


def convert_audios():
    futures = []
    executor = ProcessPoolExecutor(max_workers=16)
    for split in ['train', 'dev', 'test']:
        os.makedirs(os.path.join(tmp_dir, split, 'audios'), exist_ok=True)
        for audio_file in glob.iglob(os.path.join(tmp_dir, split, split + '_audios', '*')):
            if audio_file.split('.')[-1] not in ['mp3', 'wav']:
                continue
            if split == 'test' and not os.path.split(audio_file)[-1].startswith('squad'):
                continue
            out_file = os.path.join(tmp_dir, split, 'audios', os.path.basename(audio_file)[:-4] + '.flac')
            if os.path.exists(out_file):
                continue
            futures.append(executor.submit(partial(convert, audio_file, out_file , sr)))
            # convert(audio_file, out_file, sr)

    [f.result() for f in tqdm.tqdm(futures)]

def prepare_test_filenames():
    meta = pd.read_csv(os.path.join(tmp_dir, 'test', 'lxt_meta.csv'))
    id_maps = {}
    file_maps = {}
    for i, row in meta.iterrows():
        id = row['utterance_id']
        if id.startswith('squad'):
            d = id.split('-')
            assert len(d) == 4 and d[2] in ['q', 'c']
            if d[1] not in id_maps:
                id_maps[d[1]] = len(id_maps)
            nid = id_maps[d[1]]
            n_name = ('context' if d[2] == 'c' else 'question') + '_' + str(nid) + '_' + d[-1]
            if os.path.exists(os.path.join(tmp_dir, 'test', 'audios', n_name + '.flac')):
                os.remove(os.path.join(tmp_dir, 'test', 'audios', n_name + '.flac'))
            os.symlink(os.path.join(tmp_dir, 'test', 'audios', row['utterance_id'] + '.flac'),
                       os.path.join(tmp_dir, 'test', 'audios', n_name + '.flac'))
            file_maps[row['utterance_id']] = n_name
    json.dump(file_maps, open(os.path.join(tmp_dir, 'test', 'file_maps.json'), 'w'))


def process_data():
    samples = defaultdict(list)
    data = datasets.load_dataset('voidful/NMSQA')
    test_filemaps = json.load(open(os.path.join(tmp_dir, 'test', 'file_maps.json')))
    splits = ['test']
    for split in splits:
        for line in tqdm.tqdm(data[split]):
            if line['content_segment_audio_path'] is None:
                continue
            if line['question_audio_path'] is None:
                continue
            if split == 'test':
                if not line['content_full_audio_path'].startswith('squad'):
                    continue
                line['content_segment_audio_path'] = test_filemaps[line['content_full_audio_path'][:-4]]
                line['question_audio_path'] = test_filemaps[line['question_audio_path'][:-4]]
                line['content_full_audio_path'] = line['content_segment_audio_path'][:-len(line['content_segment_audio_path'].split('_')[-1]) - 1]
            else:
                for key in ['content_full_audio_path', 'content_segment_audio_path', 'question_audio_path']:
                    line[key] = line[key][:-4]
            if split == 'train':
                if len(line['answers']['audio_full_answer_end']) != 1 or len(line['answers']['audio_segment_answer_end']) != 1:
                    print(line)

            samples[split].append((line['question_audio_path'], line['content_segment_audio_path'],
                                   line['content_full_audio_path'],
                                   line['content_segment_normalized_text'].replace('|', ' ').replace('  ', ' '),
                                   line['question_normalized_text'].replace('|', ' ').replace('  ', ' '),
                                   line['answers']))

    meta = defaultdict(list)
    ex_meta = defaultdict(list)
    qa_pairs = defaultdict(set)
    audio_files = set()
    sdinfo = {}
    def get_info(path):
        if path not in sdinfo:
            sdinfo[path] = sd.info(path)
        return sdinfo[path]
    for split in splits:
        n_samples = total_dur = 0
        for sample in tqdm.tqdm(samples[split]):
            q_audio, s_audio, f_audio, s_text, q_text, answers = sample
            audio_path = os.path.join(tmp_dir, split, 'audios', s_audio + '.flac')
            question_path = os.path.join(tmp_dir, split, 'audios', q_audio + '.flac')
            audio_files.add((split, s_audio))
            audio_files.add((split, q_audio))
            dur = get_info(audio_path).duration + get_info(question_path).duration
            frame_question = get_info(question_path).frames
            frames = frame_question + get_info(audio_path).frames
            labels = []
            for i in range(len(answers['audio_segment_answer_end'])):
                start = int(answers['audio_segment_answer_start'][i] * sr) + frame_question
                end = int(answers['audio_segment_answer_end'][i] * sr) + frame_question
                labels.append((start, end))
            labels = sum(list(set(labels)), ())
            m = ('en' + '_' + split + '_' + q_audio.replace('_', '-') + ',' +
                 'en' + '_' + split + '_' + s_audio.replace('_', '-'), str(frames), str(frame_question),
                 q_text + '<s>' + s_text, json.dumps(labels), 'en')
            meta[split].append(m)
            qa_pairs[split + '_' + q_audio].add(split + '_' + s_audio)
            total_dur += dur
            n_samples += 1

        print("Split %s, %d samples, %.2f min" % (split, n_samples, total_dur / 60))

    for split in splits:
        n_samples = total_dur = 0

        full_buckets = defaultdict(list)
        for f in glob.iglob(os.path.join(tmp_dir, split, 'audios', '*.flac')):
            full_file = os.path.basename(f)
            full_file = full_file[:-len(full_file.split('_')[-1]) - 1]
            full_buckets[full_file].append(f)

        for sample in tqdm.tqdm(samples[split]):
            q_audio, s_audio, f_audio, s_text, q_text, answers = sample
            question_path = os.path.join(tmp_dir, split, 'audios', q_audio + '.flac')
            assert os.path.join(tmp_dir, split, 'audios', s_audio + '.flac') in full_buckets[f_audio]
            for ex_sample in full_buckets[f_audio]:
                s_audio = os.path.basename(ex_sample)[:-5]
                if split + '_' + s_audio in qa_pairs[split + '_' + q_audio]:
                    # print(sample)
                    continue
                audio_files.add((split, s_audio))
                audio_path = os.path.join(tmp_dir, split, 'audios', s_audio + '.flac')
                dur = get_info(audio_path).duration + get_info(question_path).duration
                frame_question = get_info(question_path).frames
                frames = frame_question + get_info(audio_path).frames
                m = ('en' + '_' + split + '_' + q_audio.replace('_', '-') + ',' +
                     'en' + '_' + split + '_' + s_audio.replace('_', '-'), str(frames), str(frame_question),
                     q_text + '<s>' + s_text, '[]', 'en')

                ex_meta[split].append(m)
                total_dur += dur
                n_samples += 1
        print("Ex Split %s, %d samples, %.2f min" % (split, n_samples, total_dur / 60))


    for split in splits:
        fw = open(os.path.join(out_dir, "meta.%s.txt" % split), "w", encoding='utf-8')
        for m in meta[split]:
            fw.write("|".join(m) + "\n")

        merged_meta = meta[split] + ex_meta[split]
        merged_meta.sort(key=lambda x: x[0])
        fw = open(os.path.join(out_dir, "meta.%s.ex.txt" % split), "w", encoding='utf-8')
        for m in merged_meta:
            fw.write("\t".join(m) + "\n")


    zipfiles = {}
    os.makedirs(os.path.join(out_dir, 'en'), exist_ok=True)
    for split in splits:
        zipfiles[split] = zipfile.ZipFile(os.path.join(out_dir, "en", "%s.zip" % split), 'w')

    for split, filename in tqdm.tqdm(audio_files):
        if not os.path.exists(os.path.join(tmp_dir, split, 'audios', filename + '.flac')):
            print(filename)
            continue
        zipfiles[split].write(os.path.join(tmp_dir, split, 'audios', filename + '.flac'),
                  'en' + '_' + split + '_' + filename.replace('_', '-') + '.flac')
        # shutil.copy(os.path.join(tmp_dir, split, 'audios', filename + '.flac'),
        #             os.path.join(out_dir, split, filename.replace('_', '-') + '.flac'))

def build_downsampled(filepath, out_path, k=1000):
    import random
    meta = open(filepath).read().splitlines()
    qa_segments = defaultdict(list)
    for m in meta:
        q, c = m.split('\t')[0].split(',')
        c = c[:-len(c.split('-')[-1]) - 1]
        qa_segments[(q, c)].append(m)
    qa_segments = list(qa_segments.values())
    random.seed(0)
    random.shuffle(qa_segments)
    qa_segments = qa_segments[:k]
    fw = open(out_path, 'w')
    for qa in qa_segments:
        for m in qa:
            fw.write(m + '\n')

def split_meta(meta_file, n_split=2):
    meta = open(meta_file).read().splitlines()
    questions = defaultdict(list)
    for m in meta:
        questions[m.split(',')[0]].append(m)
    print("Total %d meta, %d questions" % (len(meta), len(questions)))
    size_split = len(questions) // n_split + 1
    questions = list(questions.items())
    start = 0
    for i in range(n_split):
        k = 0
        fw = open(meta_file + '.part_%d' % i, 'w')
        for key, value in questions[start: start + size_split]:
            k += 1
            for v in value:
                fw.write(v + '\n')
        print("Part %d, %d questions" % (i, k))
        start += size_split



if __name__ == '__main__':
    pass
    convert_audios()
    prepare_test_filenames()
    process_data()
    split_meta('/temp/3690/data/nmsqa/meta.dev.ex.txt')
    build_downsampled('/temp/3690/data/nmsqa/meta.dev.ex.txt', '/temp/3690/data/nmsqa/meta.dev.ex.ds.300.txt', k=300)
