#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>; Authors of neural-lexicon-reader (See https://github.com/mutiann/neural-lexicon-reader)
#
# SPDX-FileContributor: Mutian He <mutian.he@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

from utils.hparams import HParams

hparams = HParams(
    sr=16000,

    max_eval_samples=1000,

    data_format="Iltbas",
    bucket_size=1024,
    shuffle_training_data=True,
    batch_frame_limit=1e6,
    batch_quad_frame_limit=1e11,
    batch_size=24,
    data_warmup_steps=20000,
    input_length_lower_bound=16000,
    input_length_upper_bound=160000,
    input_length_final_lower_bound=0,
    input_length_final_upper_bound=320000,
    target_length_lower_bound=2,
    use_attention_mask=True,

    pad_token_id=0,

    reg_weight=5e-3,
    max_grad_norm=1.0,
    l2_sp_weight=0.,
    use_fisher_l2sp=False,

    feat_proj_dropout=0.1,
    final_dropout=0.1,
    hidden_dropout=0.1,
    activation_dropout=0.1,
    mask_time_prob=0.1,

    warmup_steps=20000,
    plateau_steps=0,
    reset_period=0,
    reset_times=1,
    max_lr=1e-4,
    min_lr=0.,
    final_lr=3e-5,
    decay_steps=100000,
    decay_rate=1e-2,
    decay_type='inv_sqrt',
    adam_eps=1e-8,
    loss_normalize_type='sample',

    input_type='audio',
    asr_model_type='ctc',
    asr_model_name='facebook/wav2vec2-large-xlsr-53',
    asr_processor_name='',
    asr_keep_feature_encoder_only=False,
    remove_top_layers=12,
    ctc_weight=0.,
    freeze_feature_encoder=True,

    freeze_module='asr_model',
    freeze_steps=10000,
    reinit_module='.',
    use_lna=False,

    use_decoder=True,
    decoder_type='mbart',
    decoder_name='facebook/mbart-large-50-many-to-many-mmt',
    decoder_stack_encoder=True,
    decoder_remove_decoder=False,
    decoder_model_name='',
    decoder_weight=1.0,
    decoder_dropout=0.1,

    use_classifier=False,
    classifier_type='xlmr',
    classifier_name='xlm-roberta-base',
    classifier_weight=1.0,
    classifier_num_labels=200,
    classifier_head_type='classifier',
    classifier_num_targets=2,
    classifier_position='decoder_encoder',
    classifier_position_layer=0,
    classifier_pooling='max',
    classifier_dropout=0.1,
    classifier_use_projector=True,
    classifier_projector_init_std=0.006,
    classifier_hidden_size=768,
    classifier_keep_layer_from=0,
    classifier_keep_layer_to=13,

    use_infergen=False,
    infergen_type='mbart',
    infergen_name='facebook/mbart-large-50-many-to-many-mmt',
    infergen_mode='cls',
    infergen_keep_layer_from=0,
    infergen_keep_layer_to=3,
    infergen_dropout=0.1,
    infergen_attention_dropout=0.0,

    adaptor_type='conv',
    adaptor_num_layers=3,
    adaptor_pos_encoding=False,
    adaptor_pos_encoding_mode='encoding',
    adaptor_pe_weight=0.1,
    adaptor_use_layernorm=False,
    adaptor_use_glu=True,
    adaptor_output_dim=1024,

    upper_only=False,
    filter_by_charset=False,

    num_languages=20,
    use_src_lang_embed=True,
    use_tgt_lang_embed=True,
    data_groups="",
    data_group_ratio="",

    use_language_adversarial=False,
    language_adversarial_scale=1.0,

    qa_balance_factor=1.0,
    qa_label_type='sample',
    qa_segment_cls=False,
    qa_ignore_non_segment=True,
    qa_segment_cls_weight=1.0,

    eval_num_beams=5,
    eval_length_penalty=1.0,
    eval_metrics=['bleu', 'wer', 'cer', 'rouge'],
    eval_filter_samples=False,
)
