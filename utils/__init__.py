#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import torch

def dict_send_to(data, device, detach=False, as_numpy=False):
    result = {}
    for key in data:
        t = data[key]
        if isinstance(t, torch.Tensor):
            if detach:
                t = t.detach()
            t = t.to(device)
            if as_numpy:
                t = t.numpy()
        elif isinstance(t, dict):
            t = dict_send_to(t, device, detach, as_numpy)
        result[key] = t
    return result