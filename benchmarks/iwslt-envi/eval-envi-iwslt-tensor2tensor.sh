PROBLEM=translate_envi_iwslt32k
MODEL=transformer
HPARAMS=transformer_base

DATA_DIR=/nfs/project/nmt/space_vh/experiments/data/tensor2tensor/envi
TMP_DIR=/nfs/project/nmt/space_vh/tmp/envi
TRAIN_DIR=/tmp-network/vhoang/experiments/models/envi_transformer_base

# average the model checkpoints
#rm -rf $TRAIN_DIR/averaged-model/*
#CUDA_VISIBLE_DEVICES=0 python /nfs/team/nlp/users/vhoang/ve3/lib/python3.6/site-packages/tensor2tensor/utils/avg_checkpoints.py --worker_gpu=1 --prefix=$TRAIN_DIR/ --num_last_checkpoints=10 --output_path=$TRAIN_DIR/averaged-model/averaged.ckpt

# Decode
DECODE_FILE_FULL=/nfs/project/nmt/space_vh/tmp/envi/tst2013.en

BEAM_SIZE=5
ALPHA=0.6

#CUDA_VISIBLE_DEVICES=0 t2t-decoder \
# --data_dir=$DATA_DIR \
# --problems=$PROBLEM \
# --model=$MODEL \
# --hparams_set=$HPARAMS \
# --output_dir=$TRAIN_DIR/averaged-model \
# --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
# --worker_gpu=1 \
# --decode_from_file=$DECODE_FILE_FULL \
# --decode_to_file=$TRAIN_DIR/tst2013-envi.$HPARAMS.translated.vi

# See the translations

# Evaluate the BLEU score
t2t-bleu --translation=$TRAIN_DIR/tst2013-envi.$HPARAMS.translated.vi --reference=/nfs/project/nmt/space_vh/tmp/envi/tst2013.vi
/nfs/team/nlp/users/vhoang/tools/mosesdecoder/scripts/generic/multi-bleu.perl /nfs/project/nmt/space_vh/tmp/envi/tst2013.vi < $TRAIN_DIR/tst2013-envi.$HPARAMS.translated.vi > $TRAIN_DIR/tst2013-envi.$HPARAMS.translated.vi.score-BLEU
