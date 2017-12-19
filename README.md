# An Implementation of [Transformer](http://papers.nips.cc/paper/7181-attention-is-all-you-need) in [DyNet](https://github.com/clab/dynet)

### Dependencies

Before compiling dynet, you need:

 * [Eigen](https://bitbucket.org/eigen/eigen), e.g. 3.3.x

 * [cuda](https://developer.nvidia.com/cuda-toolkit) version 7.5 or higher

 * [cmake](https://cmake.org/), e.g., 3.5.1 using *cmake* ubuntu package

### Building

First, clone the repository

    git clone https://github.com/duyvuleo/Transformer-DyNet.git

As mentioned above, you'll need the latest [development] version of eigen

    hg clone https://bitbucket.org/eigen/eigen/

A modified version of latest [DyNet](https://github.com/clab/dynet) is already included (e.g., dynet folder).

#### CPU build

Compiling to execute on a CPU is as follows

    mkdir build_cpu
    cd build_cpu
    cmake .. -DEIGEN3_INCLUDE_DIR=EIGEN_PATH
    make -j 2 

MKL support. If you have Intel's MKL library installed on your machine, you can speed up the computation on the CPU by:

    cmake .. -DEIGEN3_INCLUDE_DIR=EIGEN_PATH -DMKL=TRUE -DMKL_ROOT=MKL_PATH -DENABLE_BOOST=TRUE

substituting in different paths to EIGEN_PATH and MKL_PATH if you have placed them in different directories. 

This will build the 3 binaries
    
    build_cpu/transformer-train
    build_cpu/transformer-decode

#### GPU build

Building on the GPU uses the Nvidia CUDA library, currently tested against version 7.5.
The process is as follows

    mkdir build_gpu
    cd build_gpu
    cmake .. -DBACKEND=cuda -DEIGEN3_INCLUDE_DIR=EIGEN_PATH -DCUDA_TOOLKIT_ROOT_DIR=CUDA_PATH -DCUDNN_ROOT=CUDA_PATH -DENABLE_BOOST=TRUE
    make -j 2

substituting in your EIGEN_PATH and CUDA_PATH folders, as appropriate.

This will result in the 2 binaries

    build_gpu/transformer-train
    build_gpu/transformer-decode

In general, the programs built on GPU will run much faster than on CPU (even enhanced with MKL). However, GPU is limited to the memory (8-16Gb) whereas CPU is almost unlimited. 

#### Using the model

The model can be run as follows:

    nice ./build_gpu/transformer-train --dynet-devices GPU:2 --max-seq-len 300 --minibatch-size 1024  --treport 200 --dreport 20000  --src-vocab <your-path>/data/iwslt-envi/vocab.en --tgt-vocab experiments/data/iwslt-envi/vocab.vi -t <your-path>/data/iwslt-envi/train.en-vi.vcb.capped -d experiments/data/iwslt-envi/tst2012.en-vi.vcb.capped -p <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu -e 50 --lr-eta 0.1 --lr-patience 10 --patience 20 --lr-eta-decay 2 --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 --decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.0 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 2 --num-units 128 --num-heads 2 &><your-path>/models/iwslt-envi/log.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu &

which will train a small model on a tiny training set, i.e.,

	*** DyNet initialization ***
	[dynet] initializing CUDA
	[dynet] Request for 1 specific GPU ...
	[dynet] Device Number: 2
	[dynet]   Device name: GeForce GTX TITAN X
	[dynet]   Memory Clock Rate (KHz): 3505000
	[dynet]   Memory Bus Width (bits): 384
	[dynet]   Peak Memory Bandwidth (GB/s): 336.48
	[dynet]   Memory Free (GB): 12.6801/12.8002
	[dynet]
	[dynet] Device(s) selected: 2
	[dynet] random seed: 3052832559
	[dynet] allocating memory: 512MB
	[dynet] memory allocation done.

	PID=27539
	Command: ./build_gpu/transformer-train --max-seq-len 300 --minibatch-size 1024 --treport 200 --dreport 20000 --src-vocab experiments/data/iwslt-envi/vocab.en --tgt-vocab experiments/data/iwslt-envi/vocab.vi -t experiments/data/iwslt-envi/train.en-vi.vcb.capped -d experiments/data/iwslt-envi/tst2012.en-vi.vcb.capped -p experiments/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu -e 50 --lr-eta 0.1 --lr-patience 10 --patience 20 --lr-eta-decay 2 --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 --decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.0 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 2 --num-units 128 --num-heads 2 

	Loading vocabularies from files...
	Source vocabulary file: experiments/data/iwslt-envi/vocab.en
	Target vocabulary file: experiments/data/iwslt-envi/vocab.vi
	Source vocabluary: 17191
	Target vocabluary: 7709

	Reading training data from experiments/data/iwslt-envi/train.en-vi.vcb.capped...
	133141 lines, 2963637 & 3566855 tokens (s & t), 17191 & 7709 types
	Reading dev data from experiments/data/iwslt-envi/tst2012.en-vi.vcb.capped...
	1553 lines, 31089 & 37403 tokens (s & t)

	Count of model parameters: 4194333

	Creating minibatches for training data (using minibatch_size=1024)...

	***SHUFFLE
	[lr=0.1 clips=14 updates=14] sents=228 src_unks=125 trg_unks=60 E=8.17044 ppl=3534.88 (11185.7 words/sec)
	[lr=0.1 clips=14 updates=14] sents=415 src_unks=254 trg_unks=104 E=7.7529 ppl=2328.31 (11637.7 words/sec)
	[lr=0.1 clips=7 updates=7] sents=603 src_unks=315 trg_unks=131 E=7.58751 ppl=1973.38 (11092.6 words/sec)
	[lr=0.1 clips=10 updates=10] sents=821 src_unks=401 trg_unks=170 E=7.43301 ppl=1690.89 (11051.6 words/sec)
	[lr=0.1 clips=10 updates=10] sents=1012 src_unks=501 trg_unks=214 E=7.29535 ppl=1473.43 (11139.5 words/sec)
	[lr=0.1 clips=11 updates=11] sents=1204 src_unks=590 trg_unks=244 E=7.18671 ppl=1321.75 (11217.5 words/sec)
	[lr=0.1 clips=9 updates=9] sents=1407 src_unks=669 trg_unks=283 E=7.11415 ppl=1229.24 (11182.6 words/sec)
	[lr=0.1 clips=12 updates=12] sents=1608 src_unks=783 trg_unks=327 E=7.03215 ppl=1132.46 (11279.1 words/sec)
	[lr=0.1 clips=11 updates=11] sents=1801 src_unks=871 trg_unks=356 E=6.9647 ppl=1058.59 (11325.3 words/sec)
	[lr=0.1 clips=10 updates=10] sents=2022 src_unks=965 trg_unks=387 E=6.90368 ppl=995.933 (11242.2 words/sec)
	[lr=0.1 clips=11 updates=11] sents=2200 src_unks=1060 trg_unks=433 E=6.84401 ppl=938.241 (11326.5 words/sec)
	[lr=0.1 clips=13 updates=13] sents=2412 src_unks=1163 trg_unks=534 E=6.78059 ppl=880.59 (11378 words/sec)
	[lr=0.1 clips=8 updates=8] sents=2627 src_unks=1236 trg_unks=568 E=6.74264 ppl=847.795 (11281.7 words/sec)
	[lr=0.1 clips=7 updates=7] sents=2803 src_unks=1308 trg_unks=594 E=6.71087 ppl=821.286 (11220.3 words/sec)
	...

Alternatively, you can train a large model (similar to the model configuration in the original paper) on the training set, i.e.,

    nice ./build_gpu/transformer-train --dynet-devices GPU:2 --max-seq-len 300 --minibatch-size 1024  --treport 200 --dreport 20000 --src-vocab <your-path>/data/iwslt-envi/vocab.en --tgt-vocab <your-path>/data/iwslt-envi/vocab.vi -t <your-path>/data/iwslt-envi/train.en-vi.vcb.capped -d <your-path>/data/iwslt-envi/tst2012.en-vi.vcb.capped u -p <your-path>/models/iwslt-envi/params.en-vi.transformer.h8_l6_u512_do010101010001_att1_ls00_pe1_ml300_ffrelu -e 50 --lr-eta 0.1 --lr-patience 10 --patience 20 --lr-eta-decay 2 --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 --decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.0 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 6 --num-units 512 --num-heads 8 &><your-path>/models/iwslt-envi/log.en-vi.transformer.h8_l6_u512_do010101010001_att1_ls00_pe1_ml300_ffrelu & 

For decoding/inference, we can use my integrated ensemble decoder, i.e.,

    ./build_gpu/transformer-decode --dynet-devices GPU:2 --max-seq-len 300 --src-vocab experiments/data/iwslt-envi/vocab.en --tgt-vocab experiments/data/iwslt-envi/vocab.vi --beam 5 --model-cfg experiments/models/iwslt-envi/model-small-dropout.cfg -T experiments/data/iwslt-envi/tst2013.en.vcb.capped > experiments/models/iwslt-envi/translation-beam5.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu

Decoding with n-best list will be supported very soon!

The decoding configuration file (e.g., --model-cfg experiments/models/iwslt-envi/model-small-dropout.cfg) has the following format:

    <num-units> <num-heads> <nlayers> <ff-num-units-factor> <encoder-emb-dropout> <encoder-sub-layer-dropout> <decoder-emb-dropout> <decoder-sublayer-dropout> <attention-dropout> <ff-dropout> <use-label-smoothing> <label-smoothing-weight> <position-encoding-type> <max-seq-len> <attention-type> <ff-activation-type> <your-trained-model-path>

For example, 

    128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 300 1 1 <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu
    
It's worth noting that we can have multiple models for ensemble decoding, i.e., 

    128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 300 1 1 <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu_run1
    128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 300 1 1 <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu_run2

Finally, we can evaluate the translation result with BLEU:

    <your-moses-path>/mosesdecoder-RELEASE-3.0/scripts/generic/multi-bleu.perl <your-path>/data/iwslt15-envi/tst2013.vi < <your-path>/models/iwslt-envi/translation-beam5.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu > <your-path>/models/iwslt-envi/translation-beam5.test2013.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu.score-BLEU 

## Benchmarking

### IWSLT English-Vietnamese 

	* Data for English --> Vietnamese (train: 133141; dev: 1553; test: 1268; vocab 17191 (en) & 7709 types (vn)), can be obtained from https://github.com/tensorflow/nmt. 

							BLEU (tokenized + case-sensitive)
								test2012(dev)		test2013(test)		PPLX(dev)
	- NMT (https://github.com/tensorflow/nmt)		23.8			26.1			-
	(1 biLSTM-layer encoder, 2 LSTM-layer decoders, 512 hidden/embedding dim, 512 attention dim, dropout 0.2 for attention, SGD, beam10)
	- (Luong & Manning, 2015)				-			23.3			-
	(https://nlp.stanford.edu/pubs/luong-manning-iwslt15.pdf)
	------------------------------------------------------------------------------------------------------------------
	Mantidae (https://github.com/duyvuleo/Mantidae)
	- Baseline (attentional model)				-			23.94			13.6704
	(1 bi-LSTM encoder, 2 LSTM decoders, 512 hidden/embedding dim, 512 attention dim, SGD, beam5)
		+ LSTM dropout (0.2) for encoder/decoder	-			24.96			13.0963
	------------------------------------------------------------------------------------------------------------------
	Transformer-Dynet (https://github.com/duyvuleo/Transformer-DyNet)
	- Baseline 1
	(2 heads, 2 encoder/decoder layers, 128 units, SGD, beam5)
		+ dropout (0.1)					-			23.29			12.1089 (50 epochs)
		(source and target embeddings, sub-layers (attention + feedforward))
	- Baseline 2
	(4 heads, 2 encoder/decoder layers, 256 units, SGD, beam5)
		+ dropout (0.1)					-			24.14			11.0308 (35 epochs)
		(source and target embeddings, sub-layers (attention + feedforward))
	- Baseline 3
	(8 heads, 6 encoder/decoder layers, 512 units, SGD, beam5)
		+ dropout (0.1)					-			25.30			10.5158 (28 epochs)
		(source and target embeddings, sub-layers (attention + feedforward))
	******************************************************************************************************************

### WMT17 English-German (coming soon)

## Contacts

Hoang Cong Duy Vu (vhoang2@student.unimelb.edu.au; duyvuleo@gmail.com)

---
Updated Nov 2017
