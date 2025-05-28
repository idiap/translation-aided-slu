This is the reference code for the paper [Identifying storytelling in job interviews using deep learning](https://www.sciencedirect.com/science/article/pii/S2451958825001034),
published on Computers in Human Behavior Reports. The code is based on the implementation of [The Interpreter Understands Your Meaning: 
End-to-end Spoken Language Understanding Aided by Speech Translation](https://arxiv.org/abs/2305.09652), hence
the GitHub repository is reused. Please find further updates and bugfix at 
[the repository](https://github.com/idiap/translation-aided-slu/tree/dl-interview-storytelling).

# Usage

Please prepare the dataset following our format: in the dataset directory `steadi`, there should be (meta)data files named 
like `meta.train.txt`, `meta.dev.txt`, and `meta.test.txt`, which contain the (meta)data of each sample in different partitions of the dataset, one in a line.
Each line takes the form of

```
INTERVIEW_ID.UTT_ID.START_SAMPLE.END_SAMPLE|SAMPLE_LENGTH|TEXT|ASR_TEXT|LABEL|LANGUAGE
```

For example, one of them may look like:

```
P001.001.32050.160050|128000|I: Pouvez-vous vous présenter ? CA: Oui. <s> Je m'appelle Emma et je suis étudiante en dernière année, spécialisée en économie </s> I: [hmm]|Pouvez-vous vous présenter ? Oui. <s> Je m'appelle Emma et je suis étudiante en dernière année, spécialisée en économie </s>|SD|fr
```

It corresponds to an utterance to be classified, and a segment of an audio file `P001.wav` of 16K sample rate in the same directory, within the interval of sample 320550 and 160050.
The `TEXT` field is the human transcription, and the `ASR_TEXT` field is the ASR transcription. 
When the context expansion technique is applied, `<s>` and `</s>` are used to mark the start and end of the target utterance, respectively.
The `LABEL` field is the storytelling label defined in the paper, and the `LANGUAGE` field is always `fr` in this case.

Then the training can be run, using commands like

```
python train.py --model-dir=MODEL_SAVE_DIR --data-dir=DATA_ROOT_DIR --datasets=steadi:steadi --accumulation_steps=4 --hparams=warmup_steps=500,data_warmup_steps=500,eval_filter_samples=False,input_length_upper_bound=320000,input_length_final_lower_bound=1600,input_length_final_upper_bound=480000,filter_by_charset=False,data_format=nlP_La:nltPLa,use_classifier=True,freeze_steps=0,use_decoder=False,ctc_weight=0.0,classifier_num_targets=1,use_src_lang_embed=False,use_tgt_lang_embed=False,asr_model_name=jonatasgrosman/wav2vec2-large-xlsr-53-french,asr_processor_name=jonatasgrosman/wav2vec2-large-xlsr-53-french,classifier_keep_layer_to=25,classifier_num_labels=10,classifier_pooling=sos,remove_top_layers=0,classifier_with_prompt=True,asr_model_type=none,classifier_ignore_inputs=True,adaptor_type=none,classifier_use_projector=False,max_eval_samples=3000,eval_metrics=macrof1:accuracy,max_lr=2e-5,final_lr=1e-7,reg_weight=1e-2,plateau_steps=500,classifier_name=xlm-roberta-large,classifier_hidden_size=1024,decay_type=inv_power,decay_steps=100000,classifier_max_prompt_length=2000 --eval_interval=250 --train_meta=meta.train.txt:. --eval_meta=.:meta.dev.txt --checkpoint_interval=1000 --max_steps=5000 --summary_interval=50
```
, which will run the training on the STEADI dataset, using human transcription as the training text and ASR transcription as the validation text. The data files are assumed to be saved in the directory `DATA_ROOT_DIR/steadi`, and the model will be saved in `MODEL_SAVE_DIR`.

As for the model taking audio inputs, adapted from text-input models, the command can be 
```
python train.py --model-dir=MODEL_SAVE_DIR --data-dir=DATA_ROOT_DIR --datasets=steadi:steadi --accumulation_steps=4 --hparams=warmup_steps=500,data_warmup_steps=500,eval_filter_samples=False,input_length_upper_bound=320000,input_length_final_lower_bound=1600,input_length_final_upper_bound=480000,filter_by_charset=False,data_format=nlP_La:nltPLa,use_classifier=True,freeze_steps=50000,use_decoder=False,ctc_weight=0.0,classifier_num_targets=1,use_src_lang_embed=False,use_tgt_lang_embed=False,asr_model_name=jonatasgrosman/wav2vec2-large-xlsr-53-french,asr_processor_name=jonatasgrosman/wav2vec2-large-xlsr-53-french,classifier_keep_layer_to=25,classifier_num_labels=10,classifier_pooling=sos,remove_top_layers=0,classifier_with_prompt=True,classifier_use_projector=False,max_eval_samples=3000,eval_metrics=macrof1:accuracy,max_lr=2e-5,final_lr=1e-7,reg_weight=1e-2,plateau_steps=500,classifier_name=xlm-roberta-large,classifier_hidden_size=1024,lr_step_shift=0,freeze_module=asr_model,adaptor_use_layernorm=True,adaptor_pos_encoding=True,adaptor_pos_encoding_mode=embedding,adaptor_pe_weight=1.0,decay_type=inv_power,decay_steps=100000,classifier_max_prompt_length=2000 --eval_interval=250 --train_meta=meta.train.txt:. --eval_meta=.:meta.dev.txt --checkpoint_interval=1000 --max_steps=6000 --summary_interval=50 --max_retry=10 --restore_from=TEXT_MODEL_CKPT
```

As for inferece, the command will be like
```
python infer.py --model-path=MODEL_SAVE_DIR --output-path=OUTPUT_PATH --data-dir=DATA_ROOT_DIR/steadi --eval_meta=meta.test.txt --include_steps=INFER_STEP --hparams=eval_filter_samples=False --hparams=data_format=nltPLa,eval_metrics=macrof1:accuracy
```
, which will load the hyperparameters under `MODEL_SAVE_DIR`.

Other variants of training (e.g. coalescence, expansion, ASR mixing) can be carried out by changing the data files.