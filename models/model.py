#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

import copy

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from models.build import build_asr_model, build_text_decoder, build_classifier, build_infergen, \
    Adaptor, AdversarialClassifier
from transformers.utils import ModelOutput
import logging
import traceback as tb

class Model(nn.Module):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.hparams = hparams
        self.asr_model = build_asr_model(hparams)
        if hparams.use_decoder:
            self.decoder = build_text_decoder(hparams)
        if hparams.use_decoder or hparams.use_classifier:
            self.adaptor = Adaptor(hparams, self.asr_model.config.hidden_size, hparams.adaptor_output_dim)
        if hparams.use_classifier:
            if hparams.use_infergen:
                self.infergen = build_infergen(
                    hparams, self.asr_model.config.hidden_size if hparams.classifier_position == 'encoder'
                    else hparams.adaptor_output_dim)
            else:
                self.classifier = build_classifier(
                    hparams, self.asr_model.config.hidden_size if hparams.classifier_position == 'encoder'
                    else hparams.adaptor_output_dim)
        if self.hparams.use_language_adversarial:
            self.language_adversarial = AdversarialClassifier(
                hparams, self.asr_model.config.hidden_size if hparams.classifier_position == 'encoder'
                else hparams.adaptor_output_dim, hparams.num_languages)

    def forward(self, inputs, input_masks=None, labels=None, decoder_labels=None, decoder_label_masks=None,
                src_lang=None, tgt_lang=None, generate=False, **kwargs):
        outputs = self.asr_model(input_values=inputs, attention_mask=input_masks,
                                 labels=labels if self.hparams.ctc_weight > 0 else None,
                                 output_hidden_states=True, return_dict=True)
        infer_classifier = self.hparams.use_classifier and kwargs.get('type', 's2s') == 'cls'
        if kwargs.get('type', 's2s') == 's2s' and self.hparams.use_decoder:
            infer_decoder = 'decoder'
        elif kwargs.get('type', 's2s') == 'cls':
            if self.hparams.classifier_position == 'decoder_decoder':
                infer_decoder = 'decoder'
            elif self.hparams.classifier_position == 'decoder_encoder':
                if self.hparams.classifier_position_layer == 0:
                    infer_decoder = 'adaptor'
                else:
                    infer_decoder = 'encoder'
        else:
            infer_decoder = False
        if infer_decoder in ['decoder', 'encoder'] and not self.hparams.use_decoder:
            raise ValueError('use_decoder must be True when using decoder')
        mask = self.asr_model._get_feature_vector_attention_mask(
            outputs.get('logits', outputs['hidden_states'][-1]).shape[1], input_masks)
        outputs['encoder_mask'] = mask
        if infer_decoder:
            hidden = self.adaptor(outputs['hidden_states'][-1], src_lang=src_lang, tgt_lang=tgt_lang)
            mask = self.adaptor.transform_attention_mask(mask)
            outputs['adaptor_outputs'] = hidden
            outputs['decoder_mask'] = mask
        if infer_decoder in ['decoder', 'encoder']:
            if self.hparams.decoder_stack_encoder:
                if hasattr(self.decoder, 'model'):
                    decoder_model = self.decoder.model
                else:
                    decoder_model = self.decoder
                encoder_outputs = decoder_model.encoder(inputs_embeds=hidden, attention_mask=mask,
                                                        output_hidden_states=True, return_dict=True)
                outputs['decoder_encoder_outputs'] = encoder_outputs
            else:
                encoder_outputs = hidden

        if infer_decoder == 'decoder':
            if generate:
                gen_kwargs = copy.copy(kwargs)
                for key in list(gen_kwargs.keys()):
                    if key in ['input_lengths', 'names', 'label_lengths', 'label_masks', 'dataset_names', 'type'] \
                            or key.startswith('decoder_'):
                        del gen_kwargs[key]
                if self.hparams.decoder_stack_encoder:
                    gen_kwargs['inputs_embeds'] = hidden
                else:
                    gen_kwargs['encoder_outputs'] = ModelOutput(last_hidden_state=hidden)
                decoder_outputs = self.decoder.generate(output_attentions=True, attention_mask=mask,
                                                        **gen_kwargs)
            else:
                decoder_outputs = self.decoder(attention_mask=mask,
                                               labels=decoder_labels,
                                               decoder_attention_mask=decoder_label_masks,
                                               output_hidden_states=True,
                                               return_dict=True, encoder_outputs=encoder_outputs)
            outputs['decoder_outputs'] = decoder_outputs

        if infer_classifier or self.hparams.use_language_adversarial:
            if self.hparams.classifier_position == 'decoder_encoder':
                if self.hparams.classifier_position_layer == 0:
                    classifier_feature = hidden
                else:
                    classifier_feature = encoder_outputs['hidden_states'][self.hparams.classifier_position_layer]
                classifier_mask = mask
            elif self.hparams.classifier_position == 'encoder':
                classifier_feature = outputs['hidden_states'][-1]
                classifier_mask = input_masks
            else:
                raise NotImplementedError()
            if self.hparams.use_language_adversarial:
                outputs['language_adversarial_outputs'] = self.language_adversarial(
                    classifier_feature, classifier_mask)
            if infer_classifier:
                if self.hparams.use_infergen:
                    if generate:
                        gen_kwargs = copy.copy(kwargs)
                        for key in list(gen_kwargs.keys()):
                            if key in ['input_lengths', 'names', 'label_lengths', 'label_masks', 'dataset_names',
                                       'type'] or key.startswith('decoder_'):
                                del gen_kwargs[key]
                        gen_kwargs['encoder_outputs'] = ModelOutput(last_hidden_state=classifier_feature)
                        infer_generated = self.infergen.generate(
                            output_attentions=True, attention_mask=classifier_mask, **gen_kwargs)
                        outputs['classifier_outputs'] = infer_generated[:, 1:]
                    else:
                        decoder_input_ids = torch.zeros((labels.shape[0], 1),
                                                        dtype=labels.dtype, device=labels.device)
                        decoder_input_ids = torch.cat([decoder_input_ids, labels[:, :-1]], dim=1)
                        outputs['classifier_outputs'] = self.infergen(
                            decoder_input_ids=decoder_input_ids, attention_mask=classifier_mask,
                            encoder_outputs=(classifier_feature, ))
                        outputs['classifier_outputs']['logits'] = \
                            outputs['classifier_outputs']['logits'].transpose(0, 1)
                else:
                    outputs['classifier_outputs'] = self.classifier(classifier_feature, classifier_mask)

        return outputs

def learning_rate_schedule(global_step, hp):
    if hp.reset_period > 0:
        n_reset_times = global_step // hp.reset_period
        n_reset_times = min(n_reset_times, hp.reset_times)
        global_step -= n_reset_times * hp.reset_period
    if global_step < hp.warmup_steps:
        return ((hp.max_lr - hp.min_lr) * (global_step / hp.warmup_steps) + hp.min_lr) / hp.max_lr
    elif global_step <= hp.warmup_steps + hp.plateau_steps:
        return 1
    elif global_step < hp.warmup_steps + hp.plateau_steps + hp.decay_steps:
        if hp.decay_type == 'exp':
            decay_factor = -torch.log(torch.tensor(hp.final_lr / hp.max_lr)) / hp.decay_steps
            return torch.exp(- (global_step - hp.warmup_steps - hp.plateau_steps) * decay_factor)
        elif hp.decay_type == 'linear':
            decay_factor = (hp.max_lr - hp.final_lr) / hp.decay_steps
            return 1 - (global_step - hp.warmup_steps - hp.plateau_steps) * decay_factor / hp.max_lr
        elif hp.decay_type == 'inv_sqrt':
            decay_factor = torch.sqrt(torch.tensor((hp.warmup_steps + hp.plateau_steps) / global_step))
            return decay_factor
        else:
            raise ValueError('Unknown decay type: %s' % hp.decay_type)
    else:
        return hp.final_lr / hp.max_lr


def is_weight_decayed(n):
    return n.split('.')[-1] != 'bias' and n.split('.')[-2] != 'layer_norm'

def to_long_nonneg(r):
    if isinstance(r, torch.Tensor):
        return torch.maximum(r.floor(), torch.zeros_like(r)).long()
    elif isinstance(r, np.ndarray):
        return np.maximum(np.floor(r), 0).astype(np.int64)
    else:
        return int(r)

def sample_to_frame(sample_pos, n_samples, n_frames, n_intermediate_shift=None):
    if n_intermediate_shift is not None:
        n_frames_shift = n_frames / n_intermediate_shift
        r = (sample_pos + 0.5) * (n_frames - n_frames_shift) / n_samples + n_frames_shift
    else:
        r = (sample_pos + 0.5) * n_frames / n_samples
    return to_long_nonneg(r)

def frame_to_sample(frame_pos, n_samples, n_frames, n_intermediate_shift=None):
    if n_intermediate_shift is not None:
        n_frames_shift = n_frames / n_intermediate_shift
        r = (frame_pos - n_frames_shift) * n_samples / (n_frames - n_frames_shift)
    else:
        r = (frame_pos + 0.5) * n_samples / n_frames
    return to_long_nonneg(r)

def compute_loss(batch, outputs, model, hp, global_step=None):
    if hp.ctc_weight > 0 and hp.use_decoder and hp.decoder_weight > 0 and hp.loss_normalize_type == 'samples':
        raise ValueError('Cannot normalize loss by samples when using both decoder and CTC')
    if hp.loss_normalize_type not in ['sample', 'utterance']:
        raise ValueError('Unknown loss normalization type: %s' % hp.loss_normalize_type)

    result = {}
    batch_size = batch['inputs'].size(0)
    loss = 0.
    n_frames = batch['input_lengths'].sum()

    if hp.ctc_weight > 0 and batch['type'] == 's2s':
        n_finite = (outputs['loss'] != 0).sum()
        if n_finite != batch_size:
            logging.warn('Found %d inf loss in CTC loss' % (batch_size - n_finite))
        if hp.loss_normalize_type == 'utterance':
            losses = outputs['loss'] / batch['label_lengths'].float()
            ctc_loss = losses.sum()
        elif hp.loss_normalize_type == 'sample':
            losses = outputs['loss'].clone()
            ctc_loss = outputs['loss'].sum()

        loss = loss + ctc_loss * hp.ctc_weight
        result.update({'ctc_losses': losses, 'ctc_loss': ctc_loss})

    if hp.use_decoder and isinstance(outputs.get('decoder_outputs'), dict) and batch['type'] == 's2s':
        logits = outputs['decoder_outputs']['logits']
        decoder_losses = F.cross_entropy(logits.transpose(1, 2), batch['decoder_labels'], reduction='none')
        if hp.loss_normalize_type == 'sample':
            decoder_loss = decoder_losses.sum()
        elif hp.loss_normalize_type == 'utterance':
            decoder_losses = decoder_losses.sum(-1) / batch['decoder_label_lengths'].float()
            decoder_loss = decoder_losses.sum()
        loss = loss + decoder_loss * hp.decoder_weight
        result['n_decoder_tokens'] = batch['decoder_label_lengths'].sum()
        result['decoder_losses'] = decoder_losses
        result['decoder_loss'] = decoder_loss
    
    if batch['type'] == 'cls' and isinstance(outputs.get('classifier_outputs'), dict):
        logits = outputs['classifier_outputs']['logits']
        classifier_loss = 0.
        if hp.classifier_head_type == 'classifier':
            classifier_losses = []
            for i in range(len(logits)):
                cls_loss = F.cross_entropy(logits[i], batch['labels'][:, i], reduction='none')
                classifier_loss += cls_loss.sum()
                classifier_losses.append(cls_loss)
            classifier_losses = torch.stack(classifier_losses)
            classifier_loss = classifier_loss / len(logits)
        elif hp.classifier_head_type == 'qa':
            classifier_losses = []
            for i in range(2):
                p_logits = logits[:, :, i]
                if hp.qa_label_type == 'sample':
                    output_positions = sample_to_frame(batch['labels'][:, i], batch['input_lengths'],
                                                       outputs['classifier_outputs']['lengths'],
                                                       outputs['decoder_mask'].sum(-1))
                    output_positions = torch.where(batch['labels'][:, i] == 0, 0, output_positions)
                else:
                    output_positions = batch['labels'][:, i]
                cls_loss = F.cross_entropy(p_logits, output_positions, reduction='none')
                if hp.qa_segment_cls:
                    if hp.qa_ignore_non_segment:
                        cls_loss = cls_loss * (output_positions > 0) * hp.qa_balance_factor
                else:
                    cls_loss = cls_loss * torch.where(output_positions > 0, hp.qa_balance_factor, 1 / hp.qa_balance_factor)
                classifier_loss += cls_loss.sum()
                classifier_losses.append(cls_loss)
            if hp.qa_segment_cls:
                logits = outputs['classifier_outputs']['segment_logits'].reshape([-1])
                is_segment = (batch['labels'][:, 1] > 0).float()
                cls_loss = F.binary_cross_entropy_with_logits(logits, is_segment, reduction='none')
                classifier_loss += cls_loss.sum() * hp.qa_segment_cls_weight
                classifier_losses.append(cls_loss)

            classifier_losses = torch.stack(classifier_losses).T.sum(-1)
            classifier_loss = classifier_loss / len(logits)
        loss = loss + classifier_loss * hp.classifier_weight
        result['classifier_losses'] = classifier_losses
        result['classifier_loss'] = classifier_loss

    if 'language_adversarial_outputs' in outputs:
        result['language_adversarial_losses'] = F.cross_entropy(outputs['language_adversarial_outputs'],
                                                              batch['src_lang'], reduction='none')
        result['language_adversarial_loss'] = result['language_adversarial_losses'].sum()
        if global_step is not None and global_step > 10000 and result['language_adversarial_losses'].mean() > 1:
            logging.warn('Large language adversarial loss: %.3f' % result['language_adversarial_losses'].mean())
            result['language_adversarial_loss'] = result['language_adversarial_loss'] \
                                                  / result['language_adversarial_losses'].mean().detach()
        result['language_adversarial_acc'] = (outputs['language_adversarial_outputs'].argmax(-1) == batch['src_lang']).sum()
        loss = loss + result['language_adversarial_loss']
    result['loss'] = loss
    result.update({'batch_size': batch_size, 'n_frames': n_frames})
    return result

def freeze_module(model, prefix=None, name_fn=None, frozen=True, keep_encoder_frozen=True):
    if hasattr(model, 'module'):
        model = model.module

    names = []
    if prefix is not None:
        prefix = prefix.split(':')

    for name, param in model.named_parameters():
        if (prefix is not None and any([name.startswith(p) for p in prefix]))\
                or (name_fn is not None and name_fn(name)):
            param.requires_grad = not frozen
            names.append(name)
    if names:
        if prefix is not None:
            logging.info("%s %d parameters of prefix %s:" % ('Freeze' if frozen else 'Defreeze', len(names), prefix))
        elif name_fn is not None:
            logging.info("%s %d parameters using fn" % ('Freeze' if frozen else 'Defreeze', len(names)))
        logging.info(", ".join(names))
    try:
        if keep_encoder_frozen:
            model.asr_model.freeze_feature_encoder()
    except:
        tb.print_exc()

def init_module(model, prefix):
    if hasattr(model, 'module'):
        model = model.module
    base_module = [('asr_model', model.asr_model)]
    if model.hparams.use_decoder:
        base_module.append(('decoder', model.decoder))
    for bn, m in base_module:
        for name, module in m.named_modules():
            name = bn + '.' + name
            if name.startswith(prefix):
                m._init_weights(module)

def regularize_l2_sp(model, reference, weight):
    if hasattr(model, 'module'):
        model = model.module
    loss = torch.tensor(0.0).to(list(model.parameters())[0].device)
    n_params = 0
    with torch.no_grad():
        for k, v in model.named_parameters():
            if k in reference and v.requires_grad:
                t = v - reference[k]
                if isinstance(weight, dict):
                    w = weight.get(k, 1e-5)
                else:
                    w = weight
                loss += torch.sum(w * (t ** 2)) / 2
                v.data -= w * t
                n_params += v.nelement()
    if n_params > 0:
        loss = loss / n_params
    return loss

def compute_fisher_weights_from_adam(reference, model, weight):
    groups = [[], []]
    for key in reference['param_names']:
        if is_weight_decayed(key):
            groups[0].append(key)
        else:
            groups[1].append(key)
    param_names = groups[0] + groups[1]
    missing_params = []
    weights = {}
    exist_params = dict(model.named_parameters())
    for i, key in enumerate(param_names):
        if i not in reference['optim']['state']:
            missing_params.append(key)
            continue
        if key not in exist_params:
            continue
        optim_state = reference['optim']['state'][i]
        assert optim_state['exp_avg_sq'].shape == exist_params[key].shape
        if key.startswith('module.'):
            key = key[7:]
        weights[key] = optim_state['exp_avg_sq'] * weight
        weights[key] = torch.clip(weights[key], max=1e-2)
    logging.info("Fisher diagonals collected from Adam states. Missing: " + ', '.join(missing_params))
    all_weights = torch.concat([v.reshape(-1) for v in weights.values()])
    logging.info("Diagonals mean=%.4E, min=%.4E, max=%.4E, std=%.4E" % (
        all_weights.mean(), all_weights.min(), all_weights.max(), all_weights.std()
    ))
    return weights