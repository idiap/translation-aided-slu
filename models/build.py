#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

from transformers import Wav2Vec2ForCTC, BartForConditionalGeneration, \
    T5ForConditionalGeneration, XLMRobertaModel, MBartForConditionalGeneration, LongformerModel, RobertaModel
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, AutoTokenizer
from transformers.utils import ModelOutput
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
import logging
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048, pe_weight=0.1, mode='encoding'):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_weight = pe_weight
        if mode == 'encoding':
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        elif mode == 'embedding':
            self.pe = nn.Parameter(torch.empty(max_len, d_model))
            nn.init.normal_(self.pe)

    def forward(self, x):
        x = x + self.pe_weight * self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

class Adaptor(nn.Module):
    def __init__(self, hp, in_size, out_size):
        super().__init__()
        self.hparams = hp
        self.in_size = in_size
        self.out_size = out_size
        if hp.adaptor_pos_encoding:
            self.pe = PositionalEncoding(out_size, mode=hp.adaptor_pos_encoding_mode, pe_weight=hp.adaptor_pe_weight)
        if hp.use_src_lang_embed:
            self.src_lang_embed = nn.Embedding(hp.num_languages, out_size)
            nn.init.normal_(self.src_lang_embed.weight, 0, 0.006)
        if hp.use_tgt_lang_embed:
            self.tgt_lang_embed = nn.Embedding(hp.num_languages, out_size)
            nn.init.normal_(self.tgt_lang_embed.weight, 0, 0.006)
        if hp.adaptor_type == 'conv':
            self.conv = nn.ModuleList()
            self.layernorm = nn.ModuleList()
            stride = 2
            hidden_size = in_size
            for i in range(hp.adaptor_num_layers):
                if i == hp.adaptor_num_layers - 1:
                    conv_out_size = out_size * 2 if hp.adaptor_use_glu else out_size
                else:
                    conv_out_size = hidden_size * 2 if hp.adaptor_use_glu else out_size
                self.conv.append(nn.Conv1d(hidden_size, conv_out_size, kernel_size=3, stride=stride, padding=1))
                if hp.adaptor_use_layernorm:
                    self.layernorm.append(nn.LayerNorm(hidden_size if i < hp.adaptor_num_layers - 1 else out_size))
                stride = 2
        elif hp.adaptor_type == 'linear':
            self.linear = nn.Linear(in_size, out_size, bias=False)
        elif hp.adaptor_type == 'none':
            assert in_size == out_size
        else:
            raise NotImplementedError('adaptor_type {} is not implemented'.format(self.hparams.adaptor_type))

    def forward(self, inputs, src_lang=None, tgt_lang=None):
        if self.hparams.adaptor_type == 'conv':
            inputs = inputs.transpose(1, 2)
            for i in range(self.hparams.adaptor_num_layers):
                inputs = self.conv[i](inputs)
                if self.hparams.adaptor_use_glu:
                    inputs = F.glu(inputs, dim=1)
                else:
                    inputs = F.relu(inputs)
                if self.hparams.adaptor_use_layernorm:
                    # inputs = inputs.transpose(1, 2)
                    inputs = self.layernorm[i](inputs)
            inputs = inputs.transpose(1, 2)
            if self.hparams.adaptor_pos_encoding:
                inputs = self.pe(inputs)
        elif self.hparams.adaptor_type == 'linear':
            inputs = self.linear(inputs)
        elif self.hparams.adaptor_type == 'none':
            pass
        if tgt_lang is not None and hasattr(self, 'tgt_lang_embed'):
            inputs = torch.cat([self.tgt_lang_embed(tgt_lang).unsqueeze(1), inputs], dim=1)
        if src_lang is not None and hasattr(self, 'src_lang_embed'):
            inputs = torch.cat([self.src_lang_embed(src_lang).unsqueeze(1), inputs], dim=1)
        return inputs

    def transform_attention_mask(self, mask):
        if self.hparams.adaptor_type == 'conv':
            right = mask.sum(-1) - 1
            n = mask.shape[-1] - 1
            for i in range(self.hparams.adaptor_num_layers):
                right = right // 2
                n = n // 2

            mask_ = torch.zeros(mask.shape[0], n + 1, dtype=torch.bool, device=mask.device)
            mask_[(torch.arange(mask.shape[0], device=mask_.device), right)] = True
            mask_ = mask_.flip([-1]).cumsum(-1).flip([-1]).bool()
        elif self.hparams.adaptor_type in ['linear', 'none']:
            mask_ = mask
        if hasattr(self, 'src_lang_embed'):
            mask_ = torch.cat([mask.new_ones(mask.shape[0], 1), mask_], dim=1)
        if hasattr(self, 'tgt_lang_embed'):
            mask_ = torch.cat([mask.new_ones(mask.shape[0], 1), mask_], dim=1)
        return mask_

class Identity(nn.Identity):
    def __init__(self, hidden_size=None):
        from unittest.mock import Mock
        super().__init__()
        self.config = Mock()
        self.config.hidden_size = hidden_size

    def forward(self, input_values=None, inputs=None, inputs_embeds=None, **kwargs):
        if inputs is None:
            inputs = inputs_embeds
        if inputs is None:
            inputs = input_values
        return ModelOutput(last_hidden_state=inputs, hidden_states=[inputs], **kwargs)

    def num_parameters(self):
        return 0

class Classifier(nn.Module):
    def __init__(self, hp, encoder, input_dim):
        super().__init__()
        self.hparams = hp
        self.encoder = encoder
        if hp.classifier_head_type == 'classifier':
            self._type = 'classifier'
            self.linear = nn.ModuleList([nn.Linear(hp.classifier_hidden_size, hp.classifier_num_labels)
                                         for _ in range(hp.classifier_num_targets)])
        elif hp.classifier_head_type == 'qa':
            self._type = 'qa'
            self.linear = nn.Linear(hp.classifier_hidden_size, 2)
            if hp.qa_segment_cls:
                self.segment_linear = nn.Linear(hp.classifier_hidden_size, 1)
        else:
            raise NotImplementedError('classifier_head_type {} is not implemented'.format(hp.classifier_head_type))
        if hp.classifier_use_projector:
            self.project = nn.Linear(input_dim, hp.classifier_hidden_size)
            nn.init.normal_(self.project.weight, 0, hp.classifier_projector_init_std)

    def forward(self, input_features, input_masks):
        if self.hparams.classifier_use_projector:
            input_features = self.project(input_features)
        outputs = self.encoder(inputs_embeds=input_features, attention_mask=input_masks)
        if self.hparams.classifier_pooling == 'max':
            features = outputs.last_hidden_state.max(1).values
        elif self.hparams.classifier_pooling == 'mean':
            features = outputs.last_hidden_state.mean(1)
        elif self.hparams.classifier_pooling == 'first':
            features = outputs.last_hidden_state[:, 0]
        if self._type == 'classifier':
            logits = torch.stack([self.linear[i](features) for i in range(len(self.linear))])
        elif self._type == 'qa':
            logits = self.linear(outputs.last_hidden_state) # [B, N, T, 2]
            outputs['lengths'] = input_masks.sum(-1)
            if self.hparams.qa_segment_cls:
                outputs['segment_logits'] = self.segment_linear(features) # [B, N, T, 1]
        else:
            raise NotImplementedError('classifier_head_type {} is not implemented'.format(self.hparams.classifier_head_type))
        outputs['logits'] = logits
        return outputs


class AdversarialClassifier(nn.Module):
    def __init__(self, hp, input_dim, n_target):
        super().__init__()
        self.hparams = hp
        self.project = nn.Linear(input_dim, input_dim // 2)
        self.linear = nn.Linear(input_dim // 2, n_target)

    def forward(self, input_features, input_masks):
        # input_features: [batch_size, seq_len, hidden_size]
        input_features = (input_features * input_masks.unsqueeze(-1)).sum(1) / input_masks.sum(1, keepdim=True)
        input_features = - self.hparams.language_adversarial_scale * input_features + \
                         (input_features * (1 + self.hparams.language_adversarial_scale)).detach()
        x = self.project(input_features)
        x = torch.relu(x)
        logits = self.linear(x)
        return logits

def build_asr_model(hparams):
    if hparams.asr_model_type == 'none':
        asr_model = Identity(hidden_size=hparams.adaptor_output_dim)
        asr_model._get_feature_vector_attention_mask = lambda x, y: y
        asr_model.freeze_feature_encoder = lambda: None
    else:
        if hparams.asr_model_type == 'ctc':
            asr_model = Wav2Vec2ForCTC.from_pretrained(hparams.asr_model_name,
                                                       ctc_loss_reduction="none",
                                                       pad_token_id=hparams.pad_token_id,
                                                       ctc_zero_infinity=True,
                                                       # Even we manually mask out inf, the gradients will still be nan
                                                       feat_proj_dropout=hparams.feat_proj_dropout,
                                                       final_dropout=hparams.final_dropout,
                                                       hidden_dropout=hparams.hidden_dropout,
                                                       activation_dropout=hparams.activation_dropout,
                                                       mask_time_prob=hparams.mask_time_prob,
                                                       vocab_size=300)
        else:
            raise NotImplementedError('asr_model_type {} is not implemented'.format(hparams.asr_model_type))
        asr_model.freeze_feature_encoder()

        if hparams.remove_top_layers > 0:
            asr_model.wav2vec2.encoder.layers = asr_model.wav2vec2.encoder.layers[:-hparams.remove_top_layers]

        if hparams.asr_keep_feature_encoder_only:
            asr_model.wav2vec2.encoder = Identity()

        logging.info("ASR model built, with total %.2fM parameters", asr_model.num_parameters() / 1e6)

    return asr_model

def build_text_decoder(hparams):
    if not hparams.decoder_model_name:
        hparams.decoder_model_name = hparams.decoder_name
    if hparams.decoder_type == 'bart':
        decoder = BartForConditionalGeneration.from_pretrained(hparams.decoder_model_name,
                                                               dropout=hparams.decoder_dropout,
                                                               activation_dropout=hparams.decoder_dropout)
        decoder_model = decoder.model
    elif hparams.decoder_type == 'mbart':
        decoder = MBartForConditionalGeneration.from_pretrained(hparams.decoder_model_name,
                                                                dropout=hparams.decoder_dropout,
                                                                activation_dropout=hparams.decoder_dropout)
        decoder_model = decoder.model
    elif hparams.decoder_type == 't5':
        decoder = T5ForConditionalGeneration.from_pretrained(hparams.decoder_model_name)
        decoder_model = decoder
    else:
        raise NotImplementedError('decoder_type {} is not implemented'.format(hparams.decoder_type))

    if hasattr(decoder_model, 'encoder') and not hparams.decoder_stack_encoder:
        del decoder_model.encoder
    elif hasattr(decoder_model, 'decoder') and hparams.decoder_remove_decoder:
        del decoder_model.decoder
        del decoder.lm_head
        del decoder_model.shared
        del decoder_model.encoder.embed_tokens
        pass
    decoder.config.max_length = 200
    logging.info("Text decoder built, with total %.2fM parameters", decoder.num_parameters() / 1e6)
    return decoder

def build_classifier(hparams, input_dim):
    if hparams.classifier_type == 'xlmr':
        model = XLMRobertaModel.from_pretrained(hparams.classifier_name, add_pooling_layer=False,
                                                hidden_dropout_prob=hparams.classifier_dropout,
                                                attention_probs_dropout_prob=hparams.classifier_dropout)
        model.encoder.layer = model.encoder.layer[hparams.classifier_keep_layer_from: hparams.classifier_keep_layer_to]
        del model.embeddings.word_embeddings
    elif hparams.classifier_type == 'roberta':
        model = RobertaModel.from_pretrained(hparams.classifier_name, add_pooling_layer=False,
                                                hidden_dropout_prob=hparams.classifier_dropout,
                                                attention_probs_dropout_prob=hparams.classifier_dropout)
        model.encoder.layer = model.encoder.layer[hparams.classifier_keep_layer_from: hparams.classifier_keep_layer_to]
        del model.embeddings.word_embeddings
    elif hparams.classifier_type == 'longformer':
        model = LongformerModel.from_pretrained(hparams.classifier_name,
                                                hidden_dropout_prob=hparams.classifier_dropout,
                                                attention_probs_dropout_prob=hparams.classifier_dropout)
        model.encoder.layer = model.encoder.layer[hparams.classifier_keep_layer_from: hparams.classifier_keep_layer_to]
    elif hparams.classifier_type == 'none':
        model = Identity(hidden_size=input_dim)
    else:
        raise NotImplementedError('classifier_type {} is not implemented'.format(hparams.classifier_type))
    classifier = Classifier(hparams, model, input_dim=input_dim)
    logging.info("Classifier built, with total %.2fM parameters", model.num_parameters() / 1e6)
    return classifier

def build_infergen(hparams, input_dim):
    if hparams.decoder_type == 'mbart':
        decoder = MBartForConditionalGeneration.from_pretrained(hparams.infergen_name,
                                                                dropout=hparams.infergen_dropout,
                                                                activation_dropout=hparams.infergen_dropout,
                                                                attention_dropout=hparams.infergen_attention_dropout)
    else:
        raise NotImplementedError('decoder_type {} is not implemented'.format(hparams.decoder_type))

    if hparams.infergen_mode == 'cls':
        decoder.resize_token_embeddings(hparams.classifier_num_labels)
        # nn.init.normal_(decoder.model.shared.weight, 0, 0.05)
        decoder.config.max_length = hparams.classifier_num_targets + 1
        decoder.config.forced_eos_token_id = None
        decoder.config.bos_token_id = decoder.config.pad_token_id = decoder.config.eos_token_id = decoder.config.decoder_start_token_id = 0

    del decoder.model.encoder
    decoder.model.decoder.layers = decoder.model.decoder.layers[hparams.infergen_keep_layer_from:
                                                                hparams.infergen_keep_layer_to]
    logging.info("Inference generator built, with total %.2fM parameters", decoder.num_parameters() / 1e6)
    return decoder

def build_processor(hp, vocab_path):
    processor = defaultdict(None)
    if hp.asr_model_type == 'none':
        processor['tokenizer'] = processor['extractor'] = None
    elif hp.asr_processor_name:
        processor['tokenizer'] = Wav2Vec2CTCTokenizer.from_pretrained(hp.asr_processor_name)
        processor['extractor'] = Wav2Vec2FeatureExtractor.from_pretrained(hp.asr_processor_name)
    else:
        processor['tokenizer'] = Wav2Vec2CTCTokenizer(vocab_path, unk_token='[UNK]', pad_token="[PAD]", word_delimiter_token="|")
        processor['extractor'] = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=hp.use_attention_mask)
    if hp.use_decoder:
        processor['decoder_tokenizer'] = AutoTokenizer.from_pretrained(hp.decoder_name)
    else:
        processor['decoder_tokenizer'] = None

    return processor