#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

language_id = {
    'u': 0,
    'de': 1,
    'en': 2,
    'es': 3,
    'fr': 4,
    'it': 5,
    'nl': 6,
    'pl': 7,
    'tag': 8,
    'en-sum': 9,
    'en-US': 10,
    'en-GB': 11,
    'en-AU': 12,
    'fr-FR': 13,
}

id_to_lang = {v: k for k, v in language_id.items()}

charset = {}
charset['en'] = 'ABCDEFGHIJKLMNOPQRSTUVWYXZabcdefghijklmnopqrstuvwxyz0123456789!\',-.:;? ()%/°'
charset['fr'] = charset['en'] + 'àâæçéèêëîïôœùûüÿŸÜÛÙŒÔÏÎËÊÈÉÇÆÂÀº'
charset['en-sum'] = charset['en'] + '&'