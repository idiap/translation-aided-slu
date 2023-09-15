#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import os
import torch
import logging
import sys
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean
from threading import Lock
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

lock = Lock()

def set_logger(output_path=None, name=None):
    fmt = logging.Formatter("[" + (name + ' ' if name else '') + "%(levelname)s %(asctime)s]" + " %(message)s")
    handlers = []
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    h.setLevel(logging.INFO)
    handlers.append(h)
    if output_path is not None:
        h = logging.FileHandler(output_path, 'a', 'utf-8')
        h.setFormatter(fmt)
        h.setLevel(logging.INFO)
        handlers.append(h)
    if len(logging.root.handlers) == 0:
        logging.basicConfig(handlers=handlers, level=logging.INFO)
        logging.info('logging set: ' + str(logging.root.handlers))
    else:
        logging.warn('logging is already used: ' + str(logging.root.handlers))
        while logging.root.hasHandlers():
            logging.root.removeHandler(logging.root.handlers[0])
        logging.root.setLevel(logging.INFO)
        for h in handlers:
            logging.root.addHandler(h)

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []
