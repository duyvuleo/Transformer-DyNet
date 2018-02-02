wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.de.gz
wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.en.gz
gunzip corpus.tc.de.gz
gunzip corpus.tc.en.gz
curl http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/dev.tgz | tar xvzf -

python /home/choang/da33/choang/tools/subword-nmt/learn_joint_bpe_and_vocab.py --input corpus.tc.de corpus.tc.en -s 40000 -o jbpe40K.codes --write-vocabulary jbpe40K.vocab.de jbpe40K.vocab.en

python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c jbpe40K.codes --vocabulary jbpe40K.vocab.de --vocabulary-threshold 50 < corpus.tc.de > corpus.tc.jbpe40K.de
python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c jbpe40K.codes --vocabulary jbpe40K.vocab.en --vocabulary-threshold 50 < corpus.tc.en > corpus.tc.jbpe40K.en

python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c jbpe40K.codes --vocabulary jbpe40K.vocab.de --vocabulary-threshold 50 < newstest2013.tc.de > newstest2013.tc.jbpe40K.de
python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c jbpe40K.codes --vocabulary jbpe40K.vocab.en --vocabulary-threshold 50 < newstest2013.tc.en > newstest2013.tc.jbpe40K.en
python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c jbpe40K.codes --vocabulary jbpe40K.vocab.de --vocabulary-threshold 50 < newstest2014.tc.de > newstest2014.tc.jbpe40K.de
python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c jbpe40K.codes --vocabulary jbpe40K.vocab.en --vocabulary-threshold 50 < newstest2014.tc.en > newstest2014.tc.jbpe40K.en
python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c jbpe40K.codes --vocabulary jbpe40K.vocab.de --vocabulary-threshold 50 < newstest2015.tc.de > newstest2015.tc.jbpe40K.de
python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c jbpe40K.codes --vocabulary jbpe40K.vocab.en --vocabulary-threshold 50 < newstest2015.tc.en > newstest2015.tc.jbpe40K.en
python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c jbpe40K.codes --vocabulary jbpe40K.vocab.de --vocabulary-threshold 50 < newstest2016.tc.de > newstest2016.tc.jbpe40K.de
python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c jbpe40K.codes --vocabulary jbpe40K.vocab.en --vocabulary-threshold 50 < newstest2016.tc.en > newstest2016.tc.jbpe40K.en
python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c jbpe40K.codes --vocabulary jbpe40K.vocab.de --vocabulary-threshold 50 < newstest2017.tc.de > newstest2017.tc.jbpe40K.de
python /home/choang/da33/choang/tools/subword-nmt/apply_bpe.py -c jbpe40K.codes --vocabulary jbpe40K.vocab.en --vocabulary-threshold 50 < newstest2017.tc.en > newstest2017.tc.jbpe40K.en

