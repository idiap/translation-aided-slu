#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import soundfile as sd

voices = {'fr-FR': ['fr-FR-Wavenet-A', 'fr-FR-Wavenet-B', 'fr-FR-Wavenet-C', 'fr-FR-Wavenet-D'],
          'en-US': ['en-US-Neural2-A', 'en-US-Neural2-C', 'en-US-Neural2-D',
                    'en-US-Neural2-E', 'en-US-Neural2-F', 'en-US-Neural2-G', 'en-US-Neural2-H',
                    'en-US-Neural2-I', 'en-US-Neural2-J']}

def synthesize_speech(text, lang, out_path, voice):
    from google.cloud import texttospeech
    import io

    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=lang, name=voice,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    wav, sr_ = sd.read(io.BytesIO(response.audio_content))
    sd.write(out_path, wav, sr_)

