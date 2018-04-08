# make data
python2.7 scripts/wrap-data.py en vi /nfs/project/nmt/space_vh/tmp/envi/train sample-data/tst2012 sample-data/tst2013 sample-data/vocab

# full data 
# use all available vocabularies in vocab.en and vocab.vi

# small model
mkdir experiments/models/envi/small
./build_gpu/transformer-train --dynet-devices GPU:1 --max-seq-len 150 --minibatch-size 1024  --treport 1000 --dreport 50000 --src-vocab sample-data/vocab.en --tgt-vocab sample-data/vocab.vi -t /nfs/project/nmt/space_vh/tmp/envi/train.en-vi.capped -d sample-data/tst2012.en-vi.capped -p experiments/models/envi/small -e 100 --lr-eta 0.1 --lr-patience 8 --patience 15 --lr-eta-decay 1.5 --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 --decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.1 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 2 --num-units 128 --num-heads 2 --use-label-smoothing --label-smoothing-weight 0.1 --position-encoding 2 --reset-if-stuck &>experiments/models/envi/small/train.en-vi.transformer.base_h2_l2_u128_do010101010101_att1_ls01_pe2_ml150_ffrelu.log &
# decode
./build_gpu/transformer-decode --dynet-devices GPU:1 --model-path experiments/models/envi/small --beam 5 -T sample-data/tst2013.en.capped | sed 's/<s> //g' | sed 's/ <\/s>//g' > experiments/models/envi/small/translation-beam5.en-vi.transformer.base_h2_l2_u128_do010101010101_att1_ls01_pe2_ml150_ffrelu
/nfs/team/nlp/users/vhoang/tools/mosesdecoder/scripts/generic/multi-bleu.perl sample-data/tst2013.vi < experiments/models/envi/small/translation-beam5.en-vi.transformer.base_h2_l2_u128_do010101010101_att1_ls01_pe2_ml150_ffrelu

# medium models
# run1
mkdir experiments/models/envi/run1
./build_gpu/transformer-train --dynet-devices GPU:1 --max-seq-len 150 --minibatch-size 1024  --treport 1000 --dreport 50000 --src-vocab sample-data/vocab.en --tgt-vocab sample-data/vocab.vi -t /nfs/project/nmt/space_vh/tmp/envi/train.en-vi.capped -d sample-data/tst2012.en-vi.capped -p experiments/models/envi/run1 -e 100 --lr-eta 0.1 --lr-patience 8 --patience 15 --lr-eta-decay 1.5 --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 --decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.1 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 4 --num-units 512 --num-heads 4 --use-label-smoothing --label-smoothing-weight 0.1 --position-encoding 2 --reset-if-stuck &>experiments/models/envi/run1/train.en-vi.transformer.base_h4_l4_u512_do010101010101_att1_ls01_pe2_ml150_ffrelu.log &
# decode
./build_gpu/transformer-decode --dynet-devices GPU:1 --model-path experiments/models/envi/run1 --beam 5 -T sample-data/tst2013.en.capped | sed 's/<s> //g' | sed 's/ <\/s>//g' > experiments/models/envi/run1/translation-beam5.en-vi.transformer.base_h4_l4_u512_do010101010101_att1_ls01_pe2_ml150_ffrelu
/nfs/team/nlp/users/vhoang/tools/mosesdecoder/scripts/generic/multi-bleu.perl sample-data/tst2013.vi < experiments/models/envi/run1/translation-beam5.en-vi.transformer.base_h4_l4_u512_do010101010101_att1_ls01_pe2_ml150_ffrelu

# run 2
mkdir experiments/models/envi/run2
./build_gpu/transformer-train --dynet-devices GPU:1 --max-seq-len 150 --minibatch-size 1024  --treport 1000 --dreport 50000 --src-vocab sample-data/vocab.en --tgt-vocab sample-data/vocab.vi -t /nfs/project/nmt/space_vh/tmp/envi/train.en-vi.capped -d sample-data/tst2012.en-vi.capped -p experiments/models/envi/run2 -e 100 --lr-eta 0.1 --lr-patience 8 --patience 15 --lr-eta-decay 1.5 --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 --decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.1 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 4 --num-units 512 --num-heads 4 --use-label-smoothing --label-smoothing-weight 0.1 --position-encoding 2 --reset-if-stuck &>experiments/models/envi/run2/train.en-vi.transformer.base_h4_l4_u512_do010101010101_att1_ls01_pe2_ml150_ffrelu.log &
# decode
./build_gpu/transformer-decode --dynet-devices GPU:1 --model-path experiments/models/envi/run2 --beam 5 -T sample-data/tst2013.en.capped | sed 's/<s> //g' | sed 's/ <\/s>//g' > experiments/models/envi/run2/translation-beam5.en-vi.transformer.base_h4_l4_u512_do010101010101_att1_ls01_pe2_ml150_ffrelu
/nfs/team/nlp/users/vhoang/tools/mosesdecoder/scripts/generic/multi-bleu.perl sample-data/tst2013.vi < experiments/models/envi/run2/translation-beam5.en-vi.transformer.base_h4_l4_u512_do010101010101_att1_ls01_pe2_ml150_ffrelu

# ensemble 2 runs
i
