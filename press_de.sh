TEXT=/home/sxy/Projects/CP/multi30k-wmt18/data/task1/tok/
fairseq-preprocess --source-lang en --target-lang de \
  --nwordstgt 6000 --nwordssrc 6000\
  --trainpref $TEXT/train.lc.norm.tok.bpe6000 \
  --validpref $TEXT/val.lc.norm.tok.bpe6000 \
  --testpref $TEXT/test_2017_flickr.lc.norm.tok.bpe6000 \
  --destdir data-bin-de-bpe/ \
  --dataset-impl raw