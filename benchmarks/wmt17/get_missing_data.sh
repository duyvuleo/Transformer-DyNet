#---
#!/bin/bash

#. `dirname $0`/../common/vars

src=de
tgt=en
pair=$src-$tgt

curl http://data.statmt.org/wmt17/translation-task/dev.tgz | tar xvzf -
curl http://data.statmt.org/wmt17/translation-task/test.tgz | tar xvzf -
curl http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/true.tgz | tar xvzf -

moses_scripts=/home/choang/da33/choang/tools/mosesdecoder.4.0/scripts
dev_dir=dev
test_dir=test

# dev sets
for year in 2013; do
  for lang  in $src $tgt; do
    side="src"
    if [ $lang = $tgt ]; then
      side="ref"
    fi
    $moses_scripts/ems/support/input-from-sgm.perl < $dev_dir/newstest$year-$side.$lang.sgm | \
    $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang | \
    $moses_scripts/tokenizer/tokenizer.perl -a -l $lang |  \
    $moses_scripts/recaser/truecase.perl   -model truecase-model.$lang \
    > newstest$year.tc.$lang
    
  done
done

for year in 2017; do
  for lang  in $src $tgt; do
    side="src"
    if [ $lang = $tgt ]; then
      side="ref"
    fi
    $moses_scripts/ems/support/input-from-sgm.perl < $test_dir/newstest$year-$src$tgt-$side.$lang.sgm | \
    $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang | \
    $moses_scripts/tokenizer/tokenizer.perl -a -l $lang |  \
    $moses_scripts/recaser/truecase.perl   -model truecase-model.$lang \
    > newstest$year.tc.$lang
    
  done
done

#---
