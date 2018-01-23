# abstractive summarisation (English)
mkdir experiments/models/agiga/

# BPE
mkdir experiments/data/agiga
cat /home/vhoang2/works/mantis-dev/experiments/data/agiga/train.article.txt /home/vhoang2/works/mantis-dev/experiments/data/agiga/train.title.txt | /home/vhoang2/tools/subword-nmt/learn_bpe.py -s 40000 > experiments/data/agiga/train.article-title.bpe_learned_40K #joint subword splitting

/home/vhoang2/tools/subword-nmt/apply_bpe.py -c experiments/data/agiga/train.article-title.bpe_learned_40K < /home/vhoang2/works/mantis-dev/experiments/data/agiga/train.article.txt > experiments/data/agiga/train.article.jbpe40K #train
/home/vhoang2/tools/subword-nmt/apply_bpe.py -c experiments/data/agiga/train.article-title.bpe_learned_40K < /home/vhoang2/works/mantis-dev/experiments/data/agiga/train.title.txt > experiments/data/agiga/train.title.jbpe40K

/home/vhoang2/tools/subword-nmt/apply_bpe.py -c experiments/data/agiga/train.article-title.bpe_learned_40K < /home/vhoang2/works/mantis-dev/experiments/data/agiga/valid.article.filter.txt > experiments/data/agiga/valid.filter.article.jbpe40K #dev
/home/vhoang2/tools/subword-nmt/apply_bpe.py -c experiments/data/agiga/train.article-title.bpe_learned_40K < /home/vhoang2/works/mantis-dev/experiments/data/agiga/valid.title.filter.txt > experiments/data/agiga/valid.filter.title.jbpe40K

/home/vhoang2/tools/subword-nmt/apply_bpe.py -c experiments/data/agiga/train.article-title.bpe_learned_40K < /home/vhoang2/works/mantis-dev/experiments/data/agiga/valid-6000.article.filter.txt > experiments/data/agiga/valid-6000.filter.article.jbpe40K #dev-6000
/home/vhoang2/tools/subword-nmt/apply_bpe.py -c experiments/data/agiga/train.article-title.bpe_learned_40K < /home/vhoang2/works/mantis-dev/experiments/data/agiga/valid-6000.title.filter.txt > experiments/data/agiga/valid-6000.filter.title.jbpe40K

/home/vhoang2/tools/subword-nmt/apply_bpe.py -c experiments/data/agiga/train.article-title.bpe_learned_40K < /home/vhoang2/works/mantis-dev/experiments/data/agiga/sumdata/Giga/input.txt > experiments/data/agiga/test2ksamples_agiga.article.jbpe40K #test2ksamples_agiga

/home/vhoang2/tools/subword-nmt/apply_bpe.py -c experiments/data/agiga/train.article-title.bpe_learned_40K < /home/vhoang2/works/mantis-dev/experiments/data/DUC0304/clean_2004/input.txt > experiments/data/agiga/testDUC2004.article.jbpe40K #DUC 2004

# w/ word freq cutoff 5
# *** small networks
# train
nice ./build_gpu/transformer-train --dynet-devices GPU:4  --minibatch-size 1024 --treport 1000 --dreport 100000 -t /home/vhoang2/works/mantis-dev/experiments/data/agiga/train.article-title.f5.capped -d /home/vhoang2/works/mantis-dev/experiments/data/agiga/valid-6000.article-title.filter.f5.capped -p experiments/models/agiga/params.agiga.transformer.h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu -e 100 --lr-eta 0.1 --lr-patience 8 --patience 15 --lr-eta-decay 2 --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 --decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.0 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 2 --num-units 128 --num-heads 2 --use-label-smoothing --label-smoothing-weight 0.1 --position-encoding 2 &>experiments/models/agiga/log.agiga.transformer.h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu &
# decode
# test 1 (agiga)
./build_gpu/transformer-decode --dynet-mem 5000 --dynet-devices GPU:0 --beam 5  -t /home/vhoang2/works/mantis-dev/experiments/data/agiga/train.article-title.f5.capped --model-cfg experiments/models/agiga/model-baseline-small-dropout-labelsm-sinusoid.cfg -T /home/vhoang2/works/mantis-dev/experiments/data/agiga/sumdata/Giga/input.f5.capped | sed 's/<unk>/UNK/g' | sed 's/ <\/s>//g' | sed 's/<s> //g' > experiments/models/agiga/summary-beam5.test2ksamples_agiga.transformer.h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu
# ROUGE eval
cp experiments/models/agiga/summary-beam5.test2ksamples_agiga.transformer.h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu experiments/models/agiga/test1-2ksamples/system/task1_transformer_neusum_baseline1_h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu_beam5.txt
cd experiments/models/agiga/rouge-eval-tool
sh eval.sh /home/vhoang2/works/transformer-dynet/experiments/models/agiga/test1-2ksamples/
cd ../../../..
# test 2 (DUC2004)
./build_gpu/transformer-decode --dynet-mem 5000 --dynet-devices GPU:1 --beam 5 -t /home/vhoang2/works/mantis-dev/experiments/data/agiga/train.article-title.f5.capped --model-cfg experiments/models/agiga/model-baseline-small-dropout-labelsm-sinusoid.cfg -T /home/vhoang2/works/mantis-dev/experiments/data/DUC0304/clean_2004/input.f5.capped | sed 's/<unk>/UNK/g' | sed 's/ <\/s>//g' | sed 's/<s> //g' > experiments/models/agiga/summary-beam5.test2_duc2004.transformer.h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu
# ROUGE eval
cp experiments/models/agiga/summary-beam5.test2_duc2004.transformer.h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu experiments/models/agiga/test2-DUC2004/system/task1_transformer_neusum_baseline1_h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu_beam5.txt
cd experiments/models/agiga/rouge-eval-tool
sh eval.sh /home/vhoang2/works/transformer-dynet/experiments/models/agiga/test2-DUC2004/
cd ../../../..

# *** medium networks
nice ./build_gpu/transformer-train --dynet-devices GPU:4 --minibatch-size 1024 --treport 512 --dreport 100000 -t /home/vhoang2/works/mantis-dev/experiments/data/agiga/train.article-title.f5.capped -d /home/vhoang2/works/mantis-dev/experiments/data/agiga/valid-6000.article-title.filter.f5.capped -p experiments/models/agiga/params.agiga.transformer.h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu -e 100 --lr-eta 0.1 --lr-patience 8 --patience 15 --lr-eta-decay 2 --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 --decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.0 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 4 --num-units 512 --num-heads 4 --use-label-smoothing --label-smoothing-weight 0.1 --position-encoding 2 &>experiments/models/agiga/log.agiga.transformer.h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu &
# decode
# test
./build_gpu/transformer-decode --dynet-mem 7000 --dynet-devices GPU:0 --beam 5 -t /home/vhoang2/works/mantis-dev/experiments/data/agiga/train.article-title.f5.capped --model-cfg experiments/models/agiga/model-baseline-medium-dropout-labelsm-sinusoid.cfg -T /home/vhoang2/works/mantis-dev/experiments/data/agiga/sumdata/Giga/input.f5.capped | sed 's/<unk>/UNK/g' | sed 's/ <\/s>//g' | sed 's/<s> //g' > experiments/models/agiga/summary-beam5.test2ksamples_agiga.transformer.h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu
# ROUGE eval
cp experiments/models/agiga/summary-beam5.test2ksamples_agiga.transformer.h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu experiments/models/agiga/test1-2ksamples/system/task1_transformer_neusum_baseline1_h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu_beam5.txt
cd experiments/models/agiga/rouge-eval-tool
sh eval.sh /home/vhoang2/works/transformer-dynet/experiments/models/agiga/test1-2ksamples/
cd ../../../..
# test 2 (DUC2004)
./build_gpu/transformer-decode --dynet-mem 5000 --dynet-devices GPU:5 --beam 5 -t /home/vhoang2/works/mantis-dev/experiments/data/agiga/train.article-title.f5.capped --model-cfg experiments/models/agiga/model-baseline-medium-dropout-labelsm-sinusoid.cfg -T /home/vhoang2/works/mantis-dev/experiments/data/DUC0304/clean_2004/input.f5.capped | sed 's/<unk>/UNK/g' | sed 's/ <\/s>//g' | sed 's/<s> //g' > experiments/models/agiga/summary-beam5.test2_duc2004.transformer.h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu
# ROUGE eval
cp experiments/models/agiga/summary-beam5.test2_duc2004.transformer.h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu experiments/models/agiga/test2-DUC2004/system/task1_transformer_neusum_baseline1_h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu_beam5.txt
cd experiments/models/agiga/rouge-eval-tool
sh eval.sh /home/vhoang2/works/transformer-dynet/experiments/models/agiga/test2-DUC2004/
cd ../../../..

# w/ jBPE
# *** small networks
# train
nice ./build_gpu/transformer-train --dynet-devices GPU:1 --minibatch-size 1024 --max-seq-len 100 --treport 1000 --dreport 100000 -t experiments/data/agiga/train.article-title.jbpe40K.capped -d experiments/data/agiga/valid-6000.filter.article-title.jbpe40K.capped -p experiments/models/agiga/params.agiga.transformer.h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K -e 100 --lr-eta 0.1 --lr-patience 8 --patience 15 --lr-eta-decay 2 --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 --decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.0 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 2 --num-units 128 --num-heads 2 --use-label-smoothing --label-smoothing-weight 0.1 --position-encoding 2 &>experiments/models/agiga/log.agiga.transformer.h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K &
# decode
# test 1 (agiga)
./build_gpu/transformer-decode --dynet-mem 5000 --dynet-devices GPU:0 --beam 5  -t experiments/data/agiga/train.article-title.jbpe40K.capped --model-cfg experiments/models/agiga/model-baseline-small-dropout-labelsm-sinusoid-jbpe40K.cfg -T experiments/data/agiga/test2ksamples.article.jbpe40K.capped | sed 's/<unk>/UNK/g' | sed 's/ <\/s>//g' | sed 's/<s> //g' | sed 's/@@ //g' > experiments/models/agiga/summary-beam5.test2ksamples_agiga.transformer.h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K
# ROUGE eval
cp experiments/models/agiga/summary-beam5.test2ksamples_agiga.transformer.h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K experiments/models/agiga/test1-2ksamples/system/task1_transformer_neusum_baseline1_h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K_beam5.txt
cd experiments/models/agiga/rouge-eval-tool
sh eval.sh /home/vhoang2/works/transformer-dynet/experiments/models/agiga/test1-2ksamples/
cd ../../../..
# test 2 (DUC2004)
./build_gpu/transformer-decode --dynet-mem 5000 --dynet-devices GPU:0 --beam 5 -t experiments/data/agiga/train.article-title.jbpe40K.capped --model-cfg experiments/models/agiga/model-baseline-small-dropout-labelsm-sinusoid-jbpe40K.cfg -T experiments/data/agiga/testDUC2004.article.jbpe40K.capped | sed 's/<unk>/UNK/g' | sed 's/ <\/s>//g' | sed 's/<s> //g' | sed 's/@@ //g' > experiments/models/agiga/summary-beam5.test2_duc2004.transformer.h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K
# ROUGE eval
cp experiments/models/agiga/summary-beam5.test2_duc2004.transformer.h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K experiments/models/agiga/test2-DUC2004/system/task1_transformer_neusum_baseline1_h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K_beam5.txt
cd experiments/models/agiga/rouge-eval-tool
sh eval.sh /home/vhoang2/works/transformer-dynet/experiments/models/agiga/test2-DUC2004/
cd ../../../..

# *** small networks (w/ shared source and target word embeddings)
# train
nice ./build_gpu/transformer-train --dynet-devices GPU:2 --shared-embeddings --minibatch-size 1024 --max-seq-len 100 --treport 1000 --dreport 100000 -t experiments/data/agiga/train.article-title.jbpe40K.capped -d experiments/data/agiga/valid-6000.filter.article-title.jbpe40K.capped -p experiments/models/agiga/params.agiga.transformer.h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K_shared_emb -e 100 --lr-eta 0.1 --lr-patience 8 --patience 15 --lr-eta-decay 2 --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 --decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.0 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 2 --num-units 128 --num-heads 2 --use-label-smoothing --label-smoothing-weight 0.1 --position-encoding 2 &>experiments/models/agiga/log.agiga.transformer.h2_l2_u128_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K_shared_emb &

# *** medium networks
nice ./build_gpu/transformer-train --dynet-devices GPU:3 --minibatch-size 1024 --max-seq-len 100 --treport 512 --dreport 100000 -t experiments/data/agiga/train.article-title.jbpe40K.capped -d experiments/data/agiga/valid-6000.filter.article-title.jbpe40K.capped -p experiments/models/agiga/params.agiga.transformer.h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K -e 100 --lr-eta 0.1 --lr-patience 8 --patience 15 --lr-eta-decay 2 --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 --decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.0 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 4 --num-units 512 --num-heads 4 --use-label-smoothing --label-smoothing-weight 0.1 --position-encoding 2 &>experiments/models/agiga/log.agiga.transformer.h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K &
# decode
# test 1 (agiga)
./build_gpu/transformer-decode --dynet-mem 5000 --dynet-devices GPU:0 --beam 5 -t experiments/data/agiga/train.article-title.jbpe40K.capped --model-cfg experiments/models/agiga/model-baseline-medium-dropout-labelsm-sinusoid-jbpe40K.cfg -T experiments/data/agiga/test2ksamples.article.jbpe40K.capped | sed 's/<unk>/UNK/g' | sed 's/ <\/s>//g' | sed 's/<s> //g' | sed 's/@@ //g' > experiments/models/agiga/summary-beam5.test2ksamples_agiga.transformer.h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K
# ROUGE eval
cp experiments/models/agiga/summary-beam5.test2ksamples_agiga.transformer.h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K experiments/models/agiga/test1-2ksamples/system/task1_transformer_neusum_baseline1_h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K_beam5.txt
cd experiments/models/agiga/rouge-eval-tool
sh eval.sh /home/vhoang2/works/transformer-dynet/experiments/models/agiga/test1-2ksamples/
cd ../../../..
# test 2 (DUC2004)
./build_gpu/transformer-decode --dynet-mem 5000 --dynet-devices GPU:4 --beam 5 -t experiments/data/agiga/train.article-title.jbpe40K.capped --model-cfg experiments/models/agiga/model-baseline-medium-dropout-labelsm-sinusoid-jbpe40K.cfg -T experiments/data/agiga/testDUC2004.article.jbpe40K.capped | sed 's/<unk>/UNK/g' | sed 's/ <\/s>//g' | sed 's/<s> //g' | sed 's/@@ //g' > experiments/models/agiga/summary-beam5.test2_duc2004.transformer.h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K
# ROUGE eval
cp experiments/models/agiga/summary-beam5.test2_duc2004.transformer.h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K experiments/models/agiga/test2-DUC2004/system/task1_transformer_neusum_baseline1_h4_l4_u512_do010101010001_att1_ls01_pe2_ffrelu_jbpe40K_beam5.txt
cd experiments/models/agiga/rouge-eval-tool
sh eval.sh /home/vhoang2/works/transformer-dynet/experiments/models/agiga/test2-DUC2004/
cd ../../../..

