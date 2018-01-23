# An Implementation of [Transformer](http://papers.nips.cc/paper/7181-attention-is-all-you-need) in [DyNet](https://github.com/clab/dynet)

This project aims to develop a simplified, easy-to-use implementation of Transformer architecture. However, it still has all necessary functionalities to build a complete sequence to sequence system. 

### Dependencies

Before compiling dynet, you need:

 * [Eigen](https://bitbucket.org/eigen/eigen), e.g. 3.3.x

 * [cuda](https://developer.nvidia.com/cuda-toolkit) version 7.5 or higher

 * [cmake](https://cmake.org/), e.g., 3.5.1 using *cmake* ubuntu package

### Building

First, clone the repository

    git clone https://github.com/duyvuleo/Transformer-DyNet.git

As mentioned above, you'll need the latest [development] version of eigen

    hg clone https://bitbucket.org/eigen/eigen/ (or latest stable version 3.3.4)

A modified version of latest [DyNet](https://github.com/clab/dynet) is already included (e.g., dynet folder).

#### CPU build

Compiling to execute on a CPU is as follows

    mkdir build_cpu
    cd build_cpu
    cmake .. -DEIGEN3_INCLUDE_DIR=EIGEN_PATH
    make -j 2 

MKL support. If you have Intel's MKL library installed on your machine, you can speed up the computation on the CPU by:

    cmake .. -DEIGEN3_INCLUDE_DIR=EIGEN_PATH -DMKL=TRUE -DMKL_ROOT=MKL_PATH -DENABLE_BOOST=TRUE [-DBoost_NO_BOOST_CMAKE=ON]

substituting in different paths to EIGEN_PATH and MKL_PATH if you have placed them in different directories. 

This will build the 2 binaries
    
    build_cpu/transformer-train
    build_cpu/transformer-decode

#### GPU build

Building on the GPU uses the Nvidia CUDA library, currently tested against version 7.5 and 8.0.
The process is as follows

    mkdir build_gpu
    cd build_gpu
    cmake .. -DBACKEND=cuda -DEIGEN3_INCLUDE_DIR=EIGEN_PATH -DCUDA_TOOLKIT_ROOT_DIR=CUDA_PATH -DCUDNN_ROOT=CUDA_PATH -DENABLE_BOOST=TRUE [-DBoost_NO_BOOST_CMAKE=ON]
    make -j 2

substituting in your EIGEN_PATH and CUDA_PATH folders, as appropriate.

This will result in the 2 binaries

    build_gpu/transformer-train
    build_gpu/transformer-decode

In general, the programs built on GPU will run much faster than on CPU (even enhanced with MKL). However, GPU is limited to the memory (8-16Gb) whereas CPU is almost unlimited. 

#### Using the model

The data can be processed by using the script (/scripts/wrap-data.py),

    python scripts/wrap-data.py <src-lang-id> <trg-lang-id> <train-prefix> <dev-prefix> <test-prefix> <vocab-prefix>

or

    python scripts/wrap-data.py <src-lang-id> <trg-lang-id> <train-prefix> <dev-prefix> <test-prefix> <src-freq-cutoff> <trg-freq-cutoff>

Example:

    python scripts/wrap-data.py en vi sample-data/train.10k sample-data/test2012 sample-data/test2013 vocab

or

    python scripts/wrap-data.py en vi sample-data/train.10k sample-data/test2012 sample-data/test2013 2 2

This script will create necessary data files (*.capped) that can be processed by transformer-train and transformer-decode.

First, print command line's help of transformer-train and transformer-decode,

    ./build_gpu/transformer-train --help
    ./build_gpu/transformer-decode --help

The model can be run as follows:

    nice ./build_gpu/transformer-train --dynet-devices GPU:2 --max-seq-len 300 --minibatch-size 1024  --treport 200 --dreport 20000  --src-vocab <your-path>/data/iwslt-envi/vocab.en --tgt-vocab experiments/data/iwslt-envi/vocab.vi -t <your-path>/data/iwslt-envi/train.en-vi.vcb.capped -d <your-path>/data/iwslt-envi/tst2012.en-vi.vcb.capped -p <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu -e 50 --lr-eta 0.1 --lr-patience 10 --patience 20 --lr-eta-decay 2 --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 --decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.0 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 2 --num-units 128 --num-heads 2 &><your-path>/models/iwslt-envi/log.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu &

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

For decoding/inference, we can use the integrated ensemble decoder, i.e.,

    ./build_gpu/transformer-decode --dynet-devices GPU:2 --max-seq-len 300 --src-vocab experiments/data/iwslt-envi/vocab.en --tgt-vocab experiments/data/iwslt-envi/vocab.vi --beam 5 --model-cfg experiments/models/iwslt-envi/model-small-dropout.cfg -T experiments/data/iwslt-envi/tst2013.en.vcb.capped > experiments/models/iwslt-envi/translation-beam5.test2013.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu

Decoding with n-best list can be performed, e.g.,

    ./build_gpu/transformer-decode --dynet-devices GPU:2 --max-seq-len 300 --src-vocab experiments/data/iwslt-envi/vocab.en --tgt-vocab experiments/data/iwslt-envi/vocab.vi --beam 5 --topk 100 --nbest-style moses --model-cfg experiments/models/iwslt-envi/model-small-dropout.cfg -T experiments/data/iwslt-envi/tst2013.en.vcb.capped > experiments/models/iwslt-envi/100besttranslations-beam5.test2013.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu

resulting in the following format:

    <line_number1> ||| source ||| target1 ||| TransformerModelScore=score1 || total_score1
    <line_number2> ||| source ||| target2 ||| TransformerModelScore=score2 || total_score2
    ...
	
The decoding configuration file (e.g., --model-cfg experiments/models/iwslt-envi/model-small-dropout.cfg) has the following format:

    <num-units> <num-heads> <nlayers> <ff-num-units-factor> <encoder-emb-dropout> <encoder-sub-layer-dropout> <decoder-emb-dropout> <decoder-sublayer-dropout> <attention-dropout> <ff-dropout> <use-label-smoothing> <label-smoothing-weight> <position-encoding-type> <max-seq-len> <attention-type> <ff-activation-type> <use-shared-embeddings> <use-hybrid-model> <your-trained-model-path>

For example, 

    128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 300 1 1 0 0 <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu
    
It's worth noting that we can have multiple models for ensemble decoding, i.e., 

    128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 300 1 1 0 0 <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu_run1
    128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 300 1 1 0 0 <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu_run2

Finally, we can evaluate the translation result with BLEU:

    <your-moses-path>/mosesdecoder-RELEASE-3.0/scripts/generic/multi-bleu.perl <your-path>/data/iwslt15-envi/tst2013.vi < <your-path>/models/iwslt-envi/translation-beam5.test2013.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu > <your-path>/models/iwslt-envi/translation-beam5.test2013.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu.score-BLEU 

## Benchmarks on Sequence-to-Sequence Generation Tasks

### Machine Translation

#### IWSLT English-Vietnamese 

	* Data for English --> Vietnamese (train: 133141; dev: 1553; test: 1268; vocab 17191 (en) & 7709 (vn) types), can be obtained from https://github.com/tensorflow/nmt. 

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
		w/ LSTM dropout (0.2) for encoder/decoder	-			24.96			13.0963
	------------------------------------------------------------------------------------------------------------------
	Sockeye (Transformer)					?			?			?
	(2 heads, 2 encoder/decoder layers, sinusoid positional encoding, 128 units, Adam, beam5)
	------------------------------------------------------------------------------------------------------------------
	Transformer-Dynet (https://github.com/duyvuleo/Transformer-DyNet)
	- Baseline 1a (small model)
	(2 heads, 2 encoder/decoder layers, sinusoid positional encoding, 128 units, SGD, beam5)
		w/ dropout (0.1)					
		(source and target embeddings, sub-layers (attention + feedforward))
		and label smoothing (0.1)			-			27.50			10.5622	
	- Baseline 1b (small model)
	(2 heads, 2 encoder/decoder layers, sinusoid positional encoding, 512 units, SGD, beam5)
		w/ dropout (0.1)					
		(source and target embeddings, sub-layers (attention + feedforward))
		and label smoothing (0.1)			-			27.52			9.98913					
	- Baseline 2 (medium model)
	(4 heads, 4 encoder/decoder layers, sinusoid positional encoding, 512 units, SGD, beam5)
		w/ dropout (0.1)					
		(source and target embeddings, sub-layers (attention + feedforward))
		and label smoothing (0.1)			-			27.41			9.88427
	- Ensemble (1 small and 1 medium models)		26.10			28.79			-
	- Ensemble (2 small and 2 medium models)		26.91			29.53			-
	******************************************************************************************************************

	Note/Comment: SOTA results on the task with either single or ensemble models.

#### The Kyoto Free Translation Task (English-Japanese) (updating) 

	* Data for English --> Japanese (train (clean version): 329882; dev&dev-tune: 2401; test: 1160; vocab (src & trg freq >=3, lowercased) 51159 (en) & 51626 (ja) types), can be obtained from http://www.phontron.com/kftt/#dataonly. 

							BLEU (tokenized + case-insensitive)
								dev		test		PPLX(dev&dev-tune)		Comment
	- NAIST's SMT system at KFTT 2012			21.08		23.15		-
	(KyTea/GIZA++/Moses/Lader 1.0)
	- Attentional Model (Arthur et al, 2016)		-		20.86		-				test set size reported: 1169?
	(https://arxiv.org/pdf/1606.02006.pdf)
	(4 stacked LSTMs for decoders, hidden dim 800, BiLSTM encoder with input dim 1600, Adam, beam5)
		w/ translation lexicon 	integration		-		23.20		-
	--------------------------------------------------------------------------------------------------------------------------------------------------------
	Transformer-Dynet (https://github.com/duyvuleo/Transformer-DyNet)
	- Baseline 1 (small model)
	(2 heads, 2 encoder/decoder layers, sinusoid positional encoding, 128 units, SGD, beam5)
		w/ dropout (0.1)					
		(source and target embeddings, sub-layers (attention + feedforward))
		and label smoothing (0.1)			-		20.77		15.0154					
	- Baseline 2 (medium model)
	(4 heads, 4 encoder/decoder layers, sinusoid positional encoding, 512 units, SGD, beam5)
		w/ dropout (0.1)					
		(source and target embeddings, sub-layers (attention + feedforward))
		and label smoothing (0.1)			-		23.53		13.2851	
	- Baseline 2* (same config. with baseline 2)
		w/ BPE (joint, 32K)				-		25.46		10.8828
		ensemble (2 different runs)			-		26.55		-		
	********************************************************************************************************************************************************

	Note/Comment: Single transformer model (with medium network) can outperform the best SMT (with preordering in Japanese) as well as the NMT with translation lexicon integration. Also, just simply applying joint byte-pair encoding (BPE) on both English and Japanese, we can obtain much better SOTA result on the task (25.22 vs 23.53 vs. 23.20). 

#### WMT17 English<->German (updating)

	* Data (http://www.statmt.org/wmt17/translation-task.html) (train: 5418174 after filtering 30K_jBPE-ed sentences with length >=60; dev: newstest2013), preprocessed data can be obtained from http://data.statmt.org/wmt17/translation-task/preprocessed/.

	*********************************************************************************************************************************************
	* DE-->EN (single system)
	
		                                		BLEU (tokenized + case-sensitive)
		                                        newstest2014            	newstest2015            newstest2016               newstest2017
	- Stanford NMT (Luong et al., 2015)                                     	24.9
	(top50K, UNK post replacement, 8 stacking LSTM encoder/decoder layers, 1000 hidden/embedding dim, SGD, dropout 0.2)
	------------------------------------------------------------------------------------------------------------------
	- Edinburgh NMT (Sennrich et al., 2016a)                                	26.4                    28.5
	(89500 shared BPE, ? GRU encoders, ? GRU decoders, 500 embedding dim, 1024 hidden dim, AdaDelta, beam12, pervasive dropout)
	-  (Sennrich et al., 2016b)        		29.5                    	30.4
	(back-translation with monolingual data, 89500 shared BPE, ? GRU encoders, ? GRU decoders, 620 embedding dim, 1000 hidden dim, AdaDelta, beam12)
	------------------------------------------------------------------------------------------------------------------
	- Google's NMT									29.9
	(https://github.com/tensorflow/nmt)
	(NMT + GNMT attention (beam=10), 4 LSTM encoders and decoders, 1024 units, jBPE 32K)
	------------------------------------------------------------------------------------------------------------------
	Transformer-Dynet (https://github.com/duyvuleo/Transformer-DyNet)
	- Baseline 1 (medium model)
	(4 heads, 4 encoder/decoder layers, sinusoid positional encoding, 512 units, SGD, beam5)
		w/ dropout (0.1)					
		(source and target embeddings, sub-layers (attention + feedforward))
		and label smoothing (0.1)		29.93 (10 epochs)		29.73 (10 epochs)
	*********************************************************************************************************************************************
	Note/Comment: 

	*********************************************************************************************************************************************
	* EN-->DE (single system)
	
		                                		BLEU (tokenized + case-sensitive)
		                                        newstest2014            	newstest2015            newstest2016               newstest2017
	- Google's NMT					23.7				26.5
	(https://github.com/tensorflow/nmt)
	(NMT + GNMT attention (beam=10), 4 LSTM encoders and decoders, 1024 units, jBPE 32K)
	- WMT competition (SOTA)			20.6				24.9
	(note: these BLEU scores may be with detokenisation)
	- OpenNMT (Klein et al., 2017)			19.3				-
	- tf-seq2seq (Britz et al., 2017)		22.2				25.2
	- GNMT (Wu et al., 2016)			24.6
	- Google's tensor2tensor			27.3 (SOTA)
	(original Transformer paper, Vaswani et al, 2017)
	(8 heads, 6 encoder/decoder layers, sinusoid positional encoding, 512 units, adaptive Adam, modified beam search with width 10-12, average best model)
	------------------------------------------------------------------------------------------------------------------
	Transformer-Dynet (https://github.com/duyvuleo/Transformer-DyNet)
	- Baseline 1 (medium model)
	(4 heads, 4 encoder/decoder layers, sinusoid positional encoding, 512 units, SGD, beam5)
		w/ dropout (0.1)					
		(source and target embeddings, sub-layers (attention + feedforward))
		and label smoothing (0.1)		26.18 (10 epochs)		27.82 (10 epochs)
	*********************************************************************************************************************************************
	Note/Comment: 

#### NIST English-Chinese (in plan)

## Abstractive Summarisation (updating)

	* Data for English (train (Gigaword): 3803957; dev (Gigaword, random): 6000; 2 tests: 2000 Gigaword samples and DUC 2004 test set; vocab (src & trg freq >=5): 119506 (article) & 68886 (title)  types), can be obtained from https://github.com/harvardnlp/sent-summary. Evaluation with ROUGE 1.5.5 (75-byte length limit).

	=========================================================================================================
	2000 sampled sentences from Annotated Gigaword dataset provided by Prof. Rush
		                			ROUGE-1(F1)     ROUGE-2(F1)     ROUGE-L(F1)
	- ABS+                    			29.55           11.32           26.42
	(Rush et al., 2015)
	- Pointer Net             			35.19	        16.66           32.51
	(Gulcehre et al., 2016)
	- Noisy Channel
	(Yu et al., 2017)       			34.41           16.86           31.83
	- OpenNMT                 			33.13           16.09           31.00
	(http://opennmt.net/Models/)
	-------------------------------------------------------------------
	Mantidae (https://github.com/duyvuleo/Mantidae)
	- Baseline (attentional model)			34.599          16.385          32.538 
	(deep NMT, 2 source LSTM layers, 2 target LSTM layers with dropout 0.1, 512 embedding dim, 512 hidden dim, SGD, beam5)
		w/ 100-best reranking			35.819          17.127          33.349
		(bidirectional)
	-------------------------------------------------------------------
	Transformer-Dynet (https://github.com/duyvuleo/Transformer-DyNet)
	- Baseline 1 (small model)
	(2 heads, 2 encoder/decoder layers, sinusoid positional encoding, 128 units, SGD, beam5)
		w/ dropout (0.1)			34.466		16.239		32.203
		(source and target embeddings, sub-layers (attention + feedforward))
		and label smoothing (0.1)		
			w/ BPE (40K)			34.743		16.825		32.557
	- Baseline 2 (medium model)
	(4 heads, 4 encoder/decoder layers, sinusoid positional encoding, 512 units, SGD, beam5)
		w/ dropout (0.1)			36.130		17.482		33.846
		(source and target embeddings, sub-layers (attention + feedforward))
		and label smoothing (0.1)		
			w/ BPE (40K)			35.739		17.552		33.360
	=========================================================================================================
	DUC 2004 (licensed data from LDC which needs a purchase)
	                        			ROUGE-1         ROUGE-2         ROUGE-L
	ABS+                    			28.18           8.49		23.81
	(Rush et al., 2015)
	RAS-Elman (beam10)      			28.97		8.26            24.06
	(Chopra et al., 2016) 
	-------------------------------------------------------------------
	Mantidae (https://github.com/duyvuleo/Mantidae)
	- Baseline (attentional model)			26.152          8.408           23.665
	(deep NMT, 2 source LSTM layers, 2 target LSTM layers with dropout 0.1, 512 embedding dim, 512 hidden dim, SGD, beam5)
		w/ relaxed opt				26.354          8.622           23.920
		(bidirectional) (Hoang et al., 2017)
	-------------------------------------------------------------------
	Transformer-Dynet (https://github.com/duyvuleo/Transformer-DyNet)
	- Baseline 1 (small model)
	(2 heads, 2 encoder/decoder layers, sinusoid positional encoding, 128 units, SGD, beam5)
		w/ dropout (0.1)					
		(source and target embeddings, sub-layers (attention + feedforward))
		and label smoothing (0.1)		-		-		-
			w/ BPE (40K)			27.269		9.308		24.584
	- Baseline 2 (medium model)
	(4 heads, 4 encoder/decoder layers, sinusoid positional encoding, 512 units, SGD, beam5)
		w/ dropout (0.1)			27.793		9.168		25.070
		(source and target embeddings, sub-layers (attention + feedforward))
		and label smoothing (0.1)		
			w/ BPE (40K)			28.061		9.618		25.305

## Word Ordering (coming soon)

## Sequence-to-Sequence based Dependency Parsing (English) (updating)

	* Experiments with Penn Tree Bank WSJ corpus (train: sec2-21; dev: sec22; test: sec23)

	Method							UAS		LAS		#<SHIFTs	#>SHIFTs	#ROOT_ERRORs		Note

	-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	(Methods based on transition-based neural networks)

	(Chen and Manning, 2014)				91.8		89.6		-		-		-

	StackedLSTM (Dyer et al., 2016)				93.1		90.9		-		-		-

	GloNorm (Andor et al., 2016)
		- beam 1					93.17		91.18		-		-		-
		- beam 32					94.61		92.79		-		-		-			SOTA (best reported results)
	-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	(Methods based on neural sequence to sequence learning)

	seq2seq (Wiseman et al., 2016)
		- beam 5					88.53		84.16		-		-		-
	seq2seq + BSO (Wiseman et al., 2016)
		- beam 5					91.00		87.18		-		-		-
	seq2seq + ConBSO (Wiseman et al., 2016)
		- beam 5					91.25		86.92		-		-		-

	*********
	Mantidae (https://github.com/duyvuleo/Mantidae)
	(Configuration: 1 LSTM encoder layer, 2 LSTM decoder layers, bidirectional, 512 input/hidden dims, 256 attention dim, source word frequency cutoff 2, case-sensitive in source, actions: SHIFT + LEFT_ARCs + RIGHT_ARCs with 79 types, w/ incremental training using decoder dropout 0.1 (Gal et al., 2016), SGD)
		- beam 5					91.02		88.55		7		5		3

	*********
	Transformer-Dynet (https://github.com/duyvuleo/Transformer-DyNet)
	- Baseline 1 (small model)
	(2 heads, 2 encoder/decoder layers, sinusoid positional encoding, 128 units, SGD, beam5)
		w/ dropout (0.1)					
		(source and target embeddings, sub-layers (attention + feedforward))
		and label smoothing (0.1)							
	- Baseline 2 (medium model)
	(4 heads, 4 encoder/decoder layers, sinusoid positional encoding, 512 units, SGD, beam5)
		w/ dropout (0.1)					
		(source and target embeddings, sub-layers (attention + feedforward))
		and label smoothing (0.1)			

	-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Limitation

Currently, this implementation supports single GPU only. It may be a bit slower than available toolkits (e.g., tensor2tensor, marian, sockeye). Using with multi-GPUs will be supported in the future. 

Due to limited computational resources, I only have the results for low- and medium- resource data (IWSLT, KFTT, ...). I will try my best to get some results for large-scale WMT data. 

## ToDo

1. implementation for Bahdanau attention type? (seem to be infeasible with current implementation)

2. deep FFN layers (https://arxiv.org/ftp/arxiv/papers/1712/1712.09662.pdf)

3. weighted transformer (https://arxiv.org/pdf/1711.02132.pdf)

4. average model around best checkpoints

5. adaptive learning rate following original transformer's paper

6. Other new ideas?

## Contacts

Hoang Cong Duy Vu (vhoang2@student.unimelb.edu.au; duyvuleo@gmail.com)

---
Updated Nov 2017
