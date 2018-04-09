# See what problems, models, and hyperparameter sets are available.
# You can easily swap between them (and add new ones).
#t2t-trainer --registry_help

export PYTHONPATH=/nfs/project/nmt/space_vh/works/tensor2tensor

PROBLEM=translate_envi_iwslt32k
MODEL=transformer
HPARAMS=transformer_base

DATA_DIR=/nfs/project/nmt/space_vh/experiments/data/tensor2tensor/envi
TMP_DIR=/nfs/project/nmt/space_vh/tmp/envi
TRAIN_DIR=/tmp-network/vhoang/experiments/models/envi_transformer_base

#rm -rf $DATA_DIR $TMP_DIR $TRAIN_DIR
#mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
#python -m tensor2tensor.bin.t2t_datagen \
#  --data_dir=$DATA_DIR \
#  --tmp_dir=$TMP_DIR \
#  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
CUDA_VISIBLE_DEVICES=0 python -m tensor2tensor.bin.t2t_trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --output_dir=$TRAIN_DIR \
  --train_steps=500000 \
  --hparams_set=$HPARAMS --worker_gpu=1

