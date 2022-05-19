x=
grep ^T checkpoints/$x/temp.txt | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > checkpoints/$x/test2016.ref

grep ^H checkpoints/$x/temp.txt | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > checkpoints/$x/test2016.sys

perl multi-bleu.perl checkpoints/$x/test2016.ref < checkpoints/$x/test2016.sys