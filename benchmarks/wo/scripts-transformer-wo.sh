# wo (English, PTB)
mkdir experiments/models/wo/

# *** base
mkdir experiments/models/wo/base
nice ./build_gpu/transformer-train --minibatch-size 1024 --treport 1000 --dreport 10000 --joint-vocab ~/works/mantis-dev/experiments/data/ptb/wo/train.vocab.txt -t ~/works/mantis-dev/experiments/data/ptb/wo/train.wo.capped -d ~/works/mantis-dev/experiments/data/ptb/wo/valid.wo.capped -p experiments/models/wo/base -e 100 --lr-eta 0.1 --lr-patience 8 --patience 15 --lr-eta-decay 2 --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 --decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.1 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 4 --num-units 512 --num-heads 8 --use-label-smoothing --label-smoothing-weight 0.1 --position-encoding 2 --reset-if-stuck --use-smaller-minibatch --num-resets 3 &>experiments/models/wo/base/train.log & 
# decode
# test
./build_gpu/transformer-decode --dynet-mem 5000 --beam 5 --model-path experiments/models/wo/base -T ~/works/mantis-dev/experiments/data/ptb/wo/test.inp.capped | sed 's/ <\/s>//g' | sed 's/<s> //g' > experiments/models/wo/base/ordered-beam5.test.wo.transformer.base
# mteval-13a
cd /home/vhoang2/tools/word_ordering/analysis/eval/zgen_bleu/
./ScoreBLEU.sh -t ~/works/transformer-dynet/experiments/models/wo/base/ordered-beam5.test.wo.transformer.base -r /home/vhoang2/works/mantis-dev/experiments/data/ptb/wo/test_words_no_eos.txt -odir tmp
cd ~/works/transformer-dynet
# multi-bleu
/home/vhoang2/tools/mosesdecoder-RELEASE-3.0/scripts/generic/multi-bleu.perl /home/vhoang2/works/mantis-dev/experiments/data/ptb/wo/test_words_no_eos.txt <experiments/models/wo/base/ordered-beam5.test.wo.transformer.base &>experiments/models/wo/base/ordered-beam5.test.wo.transformer.base.score-BLEU

# *** base without positional encoding in source (equivalent to bow-to-seq)
mkdir experiments/models/wo/base-no-pos_enc
nice ./build_gpu/transformer-train --minibatch-size 1024 --treport 1000 --dreport 10000 --joint-vocab ~/works/mantis-dev/experiments/data/ptb/wo/train.vocab.txt -t ~/works/mantis-dev/experiments/data/ptb/wo/train.wo.capped -d ~/works/mantis-dev/experiments/data/ptb/wo/valid.wo.capped -p experiments/models/wo/base-no-pos_enc -e 100 --lr-eta 0.1 --lr-patience 8 --patience 15 --lr-eta-decay 2 --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 --decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.1 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 4 --num-units 512 --num-heads 8 --use-label-smoothing --label-smoothing-weight 0.1 --position-encoding 2 --position-encoding-flag 2  --reset-if-stuck --use-smaller-minibatch --num-resets 3 &>experiments/models/wo/base-no-pos_enc/train.log &
# decode
# test
./build_gpu/transformer-decode --dynet-mem 5000 --beam 5 --model-path experiments/models/wo/base-no-pos_enc -T ~/works/mantis-dev/experiments/data/ptb/wo/test.inp.capped | sed 's/ <\/s>//g' | sed 's/<s> //g' > experiments/models/wo/base-no-pos_enc/ordered-beam5.test.wo.transformer.base

