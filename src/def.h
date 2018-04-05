#pragma once

// DyNet
#include "dynet/globals.h"
#include "dynet/nodes.h"
#include "dynet/param-init.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"

// STL
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>

// Boost
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

using namespace std;
using namespace dynet;

typedef int WordId;// word Id
typedef std::vector<WordId> WordIdSentence;// word Id sentence
typedef std::vector<WordIdSentence> WordIdSentences;// batches of sentences
typedef tuple<WordIdSentence, WordIdSentence> WordIdSentencePair; // Note: can be extended to include additional information (e.g., document ID)
typedef std::vector<WordIdSentencePair> WordIdCorpus;// ToDo: change to WordIdParallelCorpus?

typedef dynet::ParameterCollection DyNetModel;
typedef std::shared_ptr<DyNetModel> DyNetModelPointer;

namespace transformer {
//---
#define TRANSFORMER_RUNTIME_ASSERT(msg) do {        \
	std::ostringstream oss;                     \
	oss << "[Transformer] " << msg;             \
	throw std::runtime_error(oss.str()); }      \
	while (0);
//---

//---
#define MULTI_HEAD_ATTENTION_PARALLEL // to use pseudo-batching for multi-head attention computing (faster)
#define USE_COLWISE_DROPOUT // use col-wise dropout
#define USE_LECUN_DIST_PARAM_INIT // use Le Cun's uniform distribution for LinearLayer params initialisation (arguably faster convergence)
#define USE_KEY_QUERY_MASKINGS // use key and query maskings in multi-head attention
#define USE_LINEAR_TRANSFORMATION_BROADCASTING // use linear transformation broadcasting at final output layer (much faster)
//---

//---
enum ATTENTION_TYPE { DOT_PRODUCT=1, ADDITIVE_MLP=2 };
enum FFL_ACTIVATION_TYPE { RELU=1, SWISH=2, SWISH_LEARNABLE_BETA=3 };
//---

//---
struct SentinelMarkers{
	int _kSRC_SOS = -1;
	int _kSRC_EOS = -1;
	int _kSRC_UNK = -1;
	int _kTGT_SOS = -1;
	int _kTGT_EOS = -1;
	int _kTGT_UNK = -1;

	SentinelMarkers(){}

	SentinelMarkers(int kSRC_SOS, int kSRC_EOS, int kSRC_UNK
		, int kTGT_SOS, int kTGT_EOS, int kTGT_UNK)
	{
		_kSRC_SOS = kSRC_SOS;
		_kSRC_EOS = kSRC_EOS;
		_kSRC_UNK = kSRC_UNK;
		_kTGT_SOS = kTGT_SOS;
		_kTGT_EOS = kTGT_EOS;
		_kTGT_UNK = kTGT_UNK;
	}
};
//---

//--- 
struct ModelStats {
	unsigned _score_type = 0;// default perplexity
	double _scores[2] = {9e+99/*best so far*/, 0.f/*current*/};// If having additional score, resize this array!
	unsigned int _words_src = 0;
	unsigned int _words_tgt = 0;
	unsigned int _words_src_unk = 0;
	unsigned int _words_tgt_unk = 0;

	ModelStats(){}

	ModelStats(unsigned score_type){
		_score_type = score_type;
		if (_score_type != 0 && _score_type != 3)// BLEU/NIST/RIBES (higher is better)
		{
			_scores[0] = 0.f;
		}// else perplexity or WER (lower is better)
	}

	void update_best_score(unsigned& cpt){
		if (_score_type == 0 || _score_type == 3)// perplexity or WER (lower is better)
		{
			if (_scores[0] > _scores[1]){
				_scores[0] = _scores[1];
				cpt = 0;
			}
			else cpt++;
		}
		else{
			if (_scores[0] < _scores[1]){
				_scores[0] = _scores[1];
				cpt = 0;
			}
			else cpt++;
		}
	}

	std::string get_score_string(bool cur_or_best=true/*current*/){
		double score = cur_or_best?_scores[1]:_scores[0];
		
		std::stringstream ss;
		if (_score_type == 0){ // perplexity
			score /= _words_tgt;
			ss << "E=" << score << " PPLX=" << std::exp(score);
		}
		else
		{
			if (_score_type == 1) ss << "approxBLEU=";// approximate because it also counts for tokenization and sub-word segmentation (e.g., BPE, WP).
			else if (_score_type == 2) ss << "approxNIST=";
			else if (_score_type == 3) ss << "approxWER=";
			else if (_score_type == 4) ss << "approxRIBES=";

			ss << score;
		}
		return ss.str(); 	
	}
};
//--- 

//---
struct TransformerConfig{
	unsigned _src_vocab_size = 0;
	unsigned _tgt_vocab_size = 0;
	
	unsigned _num_units = 512;

	unsigned _nheads = 8;

	unsigned _nlayers = 6;

	unsigned _n_ff_units_factor = 4;

	bool _use_dropout = true;
	float _encoder_emb_dropout_rate = 0.1f;
	float _encoder_sublayer_dropout_rate = 0.1f;
	float _decoder_emb_dropout_rate = 0.1f;
	float _decoder_sublayer_dropout_rate = 0.1f;
	float _attention_dropout_rate = 0.1f;
	float _ff_dropout_rate = 0.1f;

	bool _use_label_smoothing = false;
	float _label_smoothing_weight = 0.1f;

	unsigned _position_encoding = 2; // 1: learned positional embedding ; 2: sinusoidal positional encoding ; 0: none
	unsigned _position_encoding_flag = 0; // 0: positional encoding applies to both encoder and decoder ; 1: for encoder only  ; 2: for decoder only
	unsigned _max_length = 500;// for learned positional embedding

	SentinelMarkers _sm;

	unsigned _attention_type = ATTENTION_TYPE::DOT_PRODUCT;

	unsigned _ffl_activation_type = FFL_ACTIVATION_TYPE::RELU;

	bool _use_hybrid_model = false;// RNN encoding over word embeddings instead of word embeddings + positional encoding

	bool _shared_embeddings = false;// use shared word embeddings between source and target

	bool _is_training = true;

	TransformerConfig(){}

	TransformerConfig(unsigned src_vocab_size
		, unsigned tgt_vocab_size
		, unsigned num_units
		, unsigned nheads
		, unsigned nlayers
		, unsigned n_ff_units_factor
		, float encoder_emb_dropout_rate
		, float encoder_sublayer_dropout_rate
		, float decoder_emb_dropout_rate
		, float decoder_sublayer_dropout_rate
		, float attention_dropout_rate
		, float ff_dropout_rate
		, bool use_label_smoothing
		, float label_smoothing_weight
		, unsigned position_encoding
		, unsigned position_encoding_flag
		, unsigned max_length
		, SentinelMarkers sm
		, unsigned attention_type
		, unsigned ffl_activation_type
		, bool shared_embeddings=false
		, bool use_hybrid_model=false
		, bool is_training=true)
	{
		_src_vocab_size = src_vocab_size;
		_tgt_vocab_size = tgt_vocab_size;
		_num_units = num_units;
		_nheads = nheads;
		_nlayers = nlayers;
		_n_ff_units_factor = n_ff_units_factor;
		_encoder_emb_dropout_rate = encoder_emb_dropout_rate;
		_encoder_sublayer_dropout_rate = encoder_sublayer_dropout_rate;
		_decoder_emb_dropout_rate = decoder_emb_dropout_rate;
		_decoder_sublayer_dropout_rate = decoder_sublayer_dropout_rate;
		_attention_dropout_rate = attention_dropout_rate;
		_ff_dropout_rate = ff_dropout_rate;
		_use_label_smoothing = use_label_smoothing;
		_label_smoothing_weight = label_smoothing_weight;
		_position_encoding = position_encoding;
		_position_encoding_flag = position_encoding_flag;
		_max_length = max_length;
		_sm = sm;
		_attention_type = attention_type;
		_ffl_activation_type = ffl_activation_type;
		_shared_embeddings = shared_embeddings;
		if (_shared_embeddings) _tgt_vocab_size = _src_vocab_size;
		_use_hybrid_model = use_hybrid_model;
		_is_training = is_training;
		_use_dropout = _is_training;
	}	

	TransformerConfig(const TransformerConfig& tfc){
		_src_vocab_size = tfc._src_vocab_size;
		_tgt_vocab_size = tfc._tgt_vocab_size;
		_num_units = tfc._num_units;
		_nheads = tfc._nheads;
		_nlayers = tfc._nlayers;
		_n_ff_units_factor = tfc._n_ff_units_factor;
		_encoder_emb_dropout_rate = tfc._encoder_emb_dropout_rate;
		_encoder_sublayer_dropout_rate = tfc._encoder_sublayer_dropout_rate;
		_decoder_emb_dropout_rate = tfc._decoder_emb_dropout_rate;
		_decoder_sublayer_dropout_rate = tfc._decoder_sublayer_dropout_rate;
		_attention_dropout_rate = tfc._attention_dropout_rate;
		_ff_dropout_rate = tfc._ff_dropout_rate;
		_use_label_smoothing = tfc._use_label_smoothing;
		_label_smoothing_weight = tfc._label_smoothing_weight;
		_position_encoding = tfc._position_encoding;
		_position_encoding_flag = tfc._position_encoding_flag;
		_max_length = tfc._max_length;
		_sm = tfc._sm;
		_attention_type = tfc._attention_type;
		_ffl_activation_type = tfc._ffl_activation_type;
		_shared_embeddings = tfc._shared_embeddings;
		_use_hybrid_model = tfc._use_hybrid_model;
		_is_training = tfc._is_training;
		_use_dropout = _is_training;
	}
};
//---

};


