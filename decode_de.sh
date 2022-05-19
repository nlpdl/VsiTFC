x=test_fr_0.0003
# x=fr/3-17/relu_0.6_step_d_0.3_ad0.1_mean_4l_head8_e256
fairseq-generate data-bin-de \
--task my_translation \
--arch baseline_attention_transformer \
--path checkpoints/$x/checkpoint46.pt \
--batch-size 64 \
--dataset-impl raw \
--remove-bpe \
--beam 5 >checkpoints/$x/temp.txt \