wget http://data.statmt.org/rsennrich/wmt16_backtranslations/de-en/news.bt.de-en.de.gz
wget http://data.statmt.org/rsennrich/wmt16_backtranslations/de-en/news.bt.de-en.en.gz
gunzip news.bt.de-en.de.gz
gunzip news.bt.de-en.en.gz

wget http://data.statmt.org/rsennrich/wmt16_backtranslations/en-de/news.bt.en-de.de.gz
wget http://data.statmt.org/rsennrich/wmt16_backtranslations/en-de/news.bt.en-de.en.gz
gunzip news.bt.en-de.de.gz
gunzip news.bt.en-de.en.gz

src=de
tgt=en

moses_scripts=/home/choang/da33/choang/tools/mosesdecoder.4.0/scripts

for lang in $src $tgt; do
    cat < news.bt.en-de.$lang | \
    $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang | \
    $moses_scripts/tokenizer/tokenizer.perl -a -l $lang |  \
    $moses_scripts/recaser/truecase.perl   -model ../truecase-model.$lang \
    > news.bt.en-de.pp.tc.$lang
done

python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c ../jbpe40K.codes --vocabulary ../jbpe40K.vocab.de --vocabulary-threshold 50 < news.bt.en-de.pp.tc.de > news.bt.en-de.pp.tc.jbpe40K.de
python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c ../jbpe40K.codes --vocabulary ../jbpe40K.vocab.en --vocabulary-threshold 50 < news.bt.en-de.pp.tc.en > news.bt.en-de.pp.tc.jbpe40K.en

for lang in $src $tgt; do
    cat < news.bt.de-en.$lang | \
    $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang | \
    $moses_scripts/tokenizer/tokenizer.perl -a -l $lang |  \
    $moses_scripts/recaser/truecase.perl   -model ../truecase-model.$lang \
    > news.bt.de-en.pp.tc.$lang
done

python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c ../jbpe40K.codes --vocabulary ../jbpe40K.vocab.de --vocabulary-threshold 50 < news.bt.de-en.pp.tc.de > news.bt.de-en.pp.tc.jbpe40K.de
python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c ../jbpe40K.codes --vocabulary ../jbpe40K.vocab.en --vocabulary-threshold 50 < news.bt.de-en.pp.tc.en > news.bt.de-en.pp.tc.jbpe40K.en

# combine with real parallel data
# DE-->EN
cat ../corpus.tc.jbpe40K.de-en.capped news.bt.de-en.pp.tc.jbpe40K.en-de.capped > corpus_edbtrans.pp.tc.jbpe40K.de-en.capped
cat ../corpus.tc.jbpe40K.en-de.capped news.bt.en-de.pp.tc.jbpe40K.en-de.capped > corpus_edbtrans.pp.tc.jbpe40K.en-de.capped

