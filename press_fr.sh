# TEXT=/home/sxy/Projects/CP/OVC-MMT/data/Multi30K_FR
# fairseq-preprocess --source-lang en --target-lang fr \
#   --nwordstgt 6000 --nwordssrc 6000\
#   --trainpref $TEXT/train.norm.tok.lc.10000bpe \
#   --validpref $TEXT/val.norm.tok.lc.10000bpe \
#   --testpref $TEXT/test.norm.tok.lc.10000bpe \
#   --destdir data-bin-fr-bpe/ \
#   --dataset-impl raw
# TEXT=/home/sxy/Projects/CP/multi30k-wmt18/data/task1/tok/
# fairseq-preprocess --source-lang en --target-lang fr \
#   --nwordstgt 6000 --nwordssrc 6000\
#   --trainpref $TEXT/train.lc.norm.tok.bpe6000 \
#   --validpref $TEXT/val.lc.norm.tok.bpe6000 \
#   --testpref $TEXT/test_2017_flickr.lc.norm.tok.bpe6000 \
#   --destdir data-bin-fr-bpe/ \
#   --dataset-impl raw

TEXT=/home/sxy/Projects/CP/work/bpe-data/
fairseq-preprocess --source-lang en --target-lang fr \
  --nwordstgt 6000 --nwordssrc 6000\
  --trainpref $TEXT/train.bpe \
  --validpref $TEXT/val.bpe \
  --testpref $TEXT/test_2017_flickr.bpe \
  --destdir data-bin-fr-gmnmt/ \
  --dataset-impl raw

