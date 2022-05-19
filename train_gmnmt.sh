fairseq-train data-bin-fr-bpe \
  --task gmnmt_translation \
  --no-progress-bar \
  --log-format simple \
  --arch new_model_gmnmt_transformer \
  --dropout 0.5 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps '1e-9' --clip-norm 0.0 \
  --warmup-updates 8000 --lr-scheduler inverse_sqrt  --lr 0.0005 \
  --max-tokens 4096 \
  --dataset-impl raw \
  --eval-bleu \
  --eval-bleu-args '{"beam":5, "max_len_a":1.2, "max_len_b":10}' \
  --eval-bleu-detok moses \
  --patience 10 \
  --num-workers 8 \
  --batch-size 64 \
  --criterion new_dy_criterion --label-smoothing 0.1  --weight-decay 0.0 \
  --save-dir checkpoints/fr/test0.75_0.7 \
  --log-file checkpoints/fr/test0.75_0.7.log \
  --best-checkpoint-metric bleu \
  --maximize-best-checkpoint-metric \
  --attention-dropout 0.1 \
  --activation-dropout 0.1 \
  --adaptive-softmax-dropout 0.1  &

  # gmnmt_matrix_no_att_cap_mat0.7_53.7_ad0.1_dropout_0.5_no_l3