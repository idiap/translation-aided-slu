This is the reference code for the paper [The Interpreter Understands Your Meaning: 
End-to-end Spoken Language Understanding Aided by Speech Translation](https://arxiv.org/abs/2305.09652), accepted by Findings of EMNLP 2023. Relevant assets including datasets will be released soon.

# Dataset preparation
Please run the corresponding scripts in corpora/ to prepare the datasets used, with the 
paths in the scripts replaced by your own ones where you downloaded and extracted the data.
For pretraining, run `covost2.py` for CoVoST2, and since MuST-C and TEDx dataset have 
the same structure, run `mustc.py` for both of them with different `db_path` and `tmp_path`.
It should be noted that as we use ASR/ST as a pretraining task, the datasets are further cleaned.
Then run `minds14.py` for MINDS-14, `slurp.py` for SLURP, and `nmsqa.py` for NMSQA.
The newly released benchmarks are provided, which are synthesized by `massive.py` for 
SLURP-Fr and `gigawords.py` for Spoken Gigawords, requiring the Google TTS environment.

# Pretraining
To pretrain the model with ST, run

`python train.py --model-dir=/temp/run/st --data-dir=/temp/data --accumulation_steps=5 --hparams=warmup_steps=10000,eval_filter_samples=True,input_length_final_lower_bound=1600,input_length_final_upper_bound=320000,filter_by_charset=True,use_src_lang_embed=False,batch_frame_limit=7e5,batch_quad_frame_limit=7e10 --src_lang=en:fr --tgt_lang=fr:en --datasets=mustc:covost2 --max_retry=100 --max_steps=130000
`

To evaluate the pretrained model on the test split of the cleaned datasets, run

`
python infer.py --model-path=/temp/run/st --output-path=/temp/run/st/test_results --data-dir=/temp/data --datasets=mustc:covost2 --hparams=eval_filter_samples=True --include_steps=120000 --eval_meta=meta.test.txt:meta.test.txt
`

Similarly, for ASR pretraining, run 

`
python train.py --model-dir=/temp/run/asr --data-dir=/temp/data --accumulation_steps=5 --hparams=warmup_steps=10000,eval_filter_samples=True,input_length_final_lower_bound=1600,input_length_final_upper_bound=320000,filter_by_charset=True,data_format=Il__as,use_src_lang_embed=False,batch_frame_limit=7e5,batch_quad_frame_limit=7e10 --src_lang=en:fr --tgt_lang=fr:en --datasets=mustc:covost2 --max_retry=100 --max_steps=130000
`

For ST+ASR pretraining, run

`
python train.py --model-dir=/temp/run/joint --data-dir=/temp/data --accumulation_steps=5 --hparams=warmup_steps=10000,eval_filter_samples=True,input_length_final_lower_bound=1600,input_length_final_upper_bound=320000,filter_by_charset=True,data_format=Il__as:Il__as:Iltbas:Iltbas,use_src_lang_embed=False,batch_frame_limit=7e5,batch_quad_frame_limit=7e10 --src_lang=en:fr --tgt_lang=fr:en --datasets=mustc:covost2:mustc:covost2 --max_retry=100 --max_steps=10000
`

# Downstream tasks
Below we list examples for commands to run experiments on downstream tasks.

For fine-tuning ST-pretraining models to SLURP,

`
python train.py --model-dir=/temp/run/slurp/st --data-dir=/temp/data/slurp_full --train_meta=/temp/data/slurp_full/meta.train.syn.txt --accumulation_steps=4 --hparams=warmup_steps=10000,eval_filter_samples=True,input_length_final_lower_bound=1600,input_length_final_upper_bound=320000,filter_by_charset=True,data_format=nltLa,use_classifier=True,freeze_module=asr_model:decoder:adaptor,freeze_steps=10000,classifier_keep_layer_from=1,classifier_keep_layer_to=4,classifier_position_layer=12,decoder_remove_decoder=True,max_eval_samples=1000,use_src_lang_embed=False,freeze_feature_encoder=False,classifier_pooling=mean --restore_from=/temp/run/st/model.ckpt-120000 --reset_training --max_retry=100
`

For joint ST-SLURP training,

`
python train.py --model-dir=/temp/run/slurp/st_joint --data-dir=/temp/data --accumulation_steps=6 --hparams=warmup_steps=10000,eval_filter_samples=True,input_length_final_lower_bound=1600,input_length_final_upper_bound=320000,filter_by_charset=True,use_classifier=True,data_format=Iltbas:Iltbas:nltLa,data_groups=2:1,classifier_keep_layer_from=1,classifier_keep_layer_to=4,classifier_position_layer=12,data_group_ratio=1:3,use_src_lang_embed=False,freeze_feature_encoder=False,classifier_pooling=mean --restore_from=/temp/run/st/model.ckpt-120000 --reset_training --max_retry=100 --src_lang=en:fr:u --tgt_lang=fr:en:u --datasets=covost2:mustc:slurp_full --train_meta=::meta.train.syn.txt
`

To test it, 

`
python infer.py --model-path=/temp/run/slurp/st_joint --output-path=/temp/run/slurp/st_joint/test_results/ --data-dir=/temp/data --datasets=slurp_full --hparams=data_format=nltLa,eval_filter_samples=False --eval_meta=meta.test.txt
`

It can be further fine-tuned on SLURP-Fr by,

`
python train.py --model-dir=/temp/run/slurp_fr/slurp_st_joint --data-dir=/temp/data/slurp_fr --accumulation_steps=4 --hparams=warmup_steps=5000,decay_steps=50000,eval_filter_samples=False,input_length_final_lower_bound=1600,input_length_final_upper_bound=320000,filter_by_charset=True,data_format=nltLa,use_classifier=True,freeze_module=asr_model:decoder:adaptor,freeze_steps=5000,classifier_keep_layer_from=1,classifier_keep_layer_to=4,classifier_position_layer=12,decoder_remove_decoder=True,use_src_lang_embed=False,freeze_feature_encoder=False,classifier_pooling=mean,data_warmup_steps=5000 --restore_from=/temp/run/slurp/st_joint/model.ckpt-150000 --reset_training --max_retry=100 --max_steps=50000 --eval_interval=1000 --checkpoint_interval=5000
`

For MINDS-14,

`
python train.py --model-dir=/temp/run/minds14/st --data-dir=/temp/data/minds14 --accumulation_steps=4 --hparams=warmup_steps=1000,decay_steps=20000,eval_filter_samples=False,input_length_final_lower_bound=1600,input_length_final_upper_bound=480000,filter_by_charset=True,data_format=nltLa,use_classifier=True,freeze_module=asr_model:decoder:adaptor,freeze_steps=1000,classifier_keep_layer_from=1,classifier_keep_layer_to=4,classifier_position_layer=12,decoder_remove_decoder=True,max_eval_samples=1000,use_src_lang_embed=False,freeze_feature_encoder=False,classifier_pooling=mean,classifier_num_targets=1,classifier_num_labels=14,data_warmup_steps=2000 --max_retry=100 --max_steps=20000 --reset_training --restore_from=/temp/run/st/model.ckpt-120000 --eval_interval=250 --checkpoint_interval=500
`

For Spoken Gigaword,

`
python train.py --model-dir=/temp/run/gigaword/st_joint --data-dir=/temp/data --src_lang=en:fr --tgt_lang=fr:en:en-sum --datasets=mustc:covost2:gigaword --accumulation_steps=5 --hparams=warmup_steps=2000,eval_filter_samples=True,input_length_final_lower_bound=1600,input_length_final_upper_bound=320000,filter_by_charset=True,data_format=Iltbas:Iltbas:Iltbas,data_groups=2:1,freeze_module=asr_model,freeze_steps=2000,max_eval_samples=1000,data_group_ratio=1:1,use_src_lang_embed=False,freeze_feature_encoder=False,batch_frame_limit=7e5,batch_quad_frame_limit=7e10,decoder_dropout=0.2,data_warmup_steps=2000 --max_retry=100 --reset_training --restore_from=/temp/run/st/model.ckpt-120000 --eval_interval=500 --checkpoint_interval=2000 --max_steps=16000 --eval_meta=.:.:meta.dev.txt
`

For NMSQA,

`
python train.py --model-dir=/temp/run/nmsqa/st --data-dir=/temp/data --datasets=nmsqa --accumulation_steps=2 --hparams=warmup_steps=5000,eval_filter_samples=False,input_length_final_lower_bound=1600,input_length_final_upper_bound=1440000,filter_by_charset=True,use_classifier=True,data_format=DlStLa,freeze_module=asr_model:adaptor:decoder.model.encoder.layers.0.:decoder.model.encoder.layers.1.:decoder.model.encoder.layers.2.:decoder.model.encoder.layers.3.:decoder.model.encoder.layers.4.:decoder.model.encoder.layers.5.:decoder.model.encoder.layers.6.,freeze_steps=200000,decoder_remove_decoder=True,classifier_keep_layer_from=1,classifier_keep_layer_to=4,classifier_position_layer=12,use_src_lang_embed=False,freeze_feature_encoder=False,batch_size=32,batch_frame_limit=6e6,batch_quad_frame_limit=6e11,classifier_type=longformer,classifier_name=allenai/longformer-large-4096,classifier_head_type=qa,classifier_hidden_size=1024,data_warmup_steps=5000,qa_segment_cls=True,qa_balance_factor=1.0,qa_label_type=sample --reset_training --restore_from=/temp/run/st/model.ckpt-120000 --eval_meta=meta.dev.ex.ds.300.txt --train_meta=meta.train.ex.txt --checkpoint_interval=2000 --eval_interval=2000 --max_retry=100
`
, which only evaluate on part of the dev split each time for efficiency. For full evaluation on dev,

`
python infer.py --model-path=/temp/run/nmsqa/st --output-path=/temp/run/nmsqa/st/dev_results/ --data-dir=/temp/data --datasets=nmsqa --eval_meta=meta.dev.ex.txt
`
