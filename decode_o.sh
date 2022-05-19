x=new_model_bpe_alpha0.6_53.2_true_encoder_to_decoder_no_relate_gate_l1cross_l3
# x=fr/3-17/relu_0.6_step_d_0.3_ad0.1_mean_4l_head8_e256
fairseq-generate data-bin-fr-bpe \
--task new_translation \
--arch new_model_transformer \
--path checkpoints/$x/checkpoint73.pt \
--batch-size 64 \
--dataset-impl raw \
--remove-bpe \
--beam 5 >checkpoints/$x/temp.txt \