/* This is an implementation of Transformer architecture from https://arxiv.org/abs/1706.03762 (Attention is All You need).
* Developed by Cong Duy Vu Hoang
* Updated: 1 Nov 2017
*/

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

// Others
#include "def.h"
#include "str-utils.h"
#include "dict-utils.h"
#include "expr-xtra.h"
#include "math-utils.h"

using namespace std;
using namespace dynet;

typedef dynet::ParameterCollection DyNetModel;
typedef std::shared_ptr<DyNetModel> DyNetModelPointer;

namespace transformer {

enum ATTENTION_TYPE { DOT_PRODUCT = 1, ADDITIVE_MLP = 2 };

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
struct TransformerConfig{
	unsigned _src_vocab_size = 0;
	unsigned _tgt_vocab_size = 0;
	
	unsigned _num_units = 512;

	unsigned _nheads = 8;

	unsigned _nlayers = 6;

	float _dropout_rate = 0.1f;

	bool _use_label_smoothing = false;

	unsigned _position_encoding = 1; // 1: learned positional embedding ; 2: sinusoidal positional encoding ; 0: none
	unsigned _max_length = 500;// for learned positional embedding

	SentinelMarkers _sm;

	unsigned _attention_type = ATTENTION_TYPE::DOT_PRODUCT;

	TransformerConfig(){}

	TransformerConfig(unsigned src_vocab_size
		, unsigned tgt_vocab_size
		, unsigned num_units
		, unsigned nheads
		, unsigned nlayers
		, float dropout_rate
		, bool use_label_smoothing
		, unsigned position_encoding
		, unsigned max_length
		, SentinelMarkers sm
		, unsigned attention_type)
	{
		_src_vocab_size = src_vocab_size;
		_tgt_vocab_size = tgt_vocab_size;
		_num_units = num_units;
		_nheads = nheads;
		_nlayers = nlayers;
		_dropout_rate = dropout_rate;
		_use_label_smoothing = use_label_smoothing;
		_position_encoding = position_encoding;
		_max_length = max_length;
		_sm = sm;
		_attention_type = attention_type;
	}

	TransformerConfig(const TransformerConfig& tfc){
		_src_vocab_size = tfc._src_vocab_size;
		_tgt_vocab_size = tfc._tgt_vocab_size;
		_num_units = tfc._num_units;
		_nheads = tfc._nheads;
		_nlayers = tfc._nlayers;
		_dropout_rate = tfc._dropout_rate;
		_use_label_smoothing = tfc._use_label_smoothing;
		_position_encoding = tfc._position_encoding;
		_max_length = tfc._max_length;
		_sm = tfc._sm;
		_attention_type = tfc._attention_type;
	}
};
//---

//---
struct ConvLayer{
	explicit ConvLayer(DyNetModel* mod, unsigned units_size, unsigned filter_size)
		: _stride({1, 1})
	{
		_p_W = mod->add_parameters({1/*H_SIZE*/, 1/*W_SIZE*/, units_size, filter_size});
		_p_b = mod->add_parameters({filter_size});
	}

	dynet::Expression apply(dynet::ComputationGraph& cg, const dynet::Expression& i_x){
		dynet::Expression i_W = dynet::parameter(cg, _p_W);
		dynet::Expression i_b = dynet::parameter(cg, _p_b);

		dynet::Expression i_x_out = dynet::conv2d(i_x, i_W, i_b, _stride, false);		
		return i_x_out;
	}

	~ConvLayer(){}

	dynet::Parameter _p_W;
	dynet::Parameter _p_b;

	std::vector<unsigned> _stride;
};

struct FeedForwardLayer{
	explicit FeedForwardLayer(DyNetModel* mod, const TransformerConfig& tfc)
		: _innerConv(mod, tfc._num_units, tfc._num_units * 4)
		, _outerConv(mod, tfc._num_units * 4, tfc._num_units)
	{		
	}	

	~FeedForwardLayer(){}	

	dynet::Expression build_graph(dynet::ComputationGraph& cg, const dynet::Expression& i_inp/*num_units x L*/){
		// FFN(x) = relu(x * W1 + b1) * W2 + b2
		dynet::Expression i_inner = dynet::reshape(i_inp, {1, i_inp.dim().d[1], i_inp.dim().d[0]});
		i_inner = _innerConv.apply(cg, i_inner);// x * W1 + b1
		i_inner = dynet::rectify(i_inner);// relu
		dynet::Expression i_outer = _outerConv.apply(cg, i_inner);// relu(x * W1 + b1) * W2 + b2
		i_outer = dynet::reshape(i_outer, {i_inp.dim().d[0], i_inp.dim().d[1]});

		return i_outer;
	}

	ConvLayer _innerConv;
	ConvLayer _outerConv;
};
//---

//---
struct MultiHeadAttentionLayer{
	explicit MultiHeadAttentionLayer(DyNetModel* mod, const TransformerConfig& tfc, bool is_self_attention = true)
		: _attention_type(tfc._attention_type), _nheads(tfc._nheads)
	{
		_p_WQ.resize(tfc._nheads);
		_p_WK.resize(tfc._nheads);
		_p_WV.resize(tfc._nheads);
		for (unsigned h = 0; h < tfc._nheads; h++){
			_p_WQ[h] = mod->add_parameters({tfc._num_units, tfc._num_units / tfc._nheads});// dk = tfc._num_units / tfc._nheads
			_p_WK[h] = mod->add_parameters({tfc._num_units, tfc._num_units / tfc._nheads});// dk
			_p_WV[h] = mod->add_parameters({tfc._num_units, tfc._num_units / tfc._nheads});// dv = tfc._num_units / tfc._nheads
		}

		_p_WO = mod->add_parameters({tfc._num_units, tfc._num_units});
		
		_att_scale = sqrt(tfc._num_units / tfc._nheads);
	}

	~MultiHeadAttentionLayer(){}

	dynet::Expression build_graph(dynet::ComputationGraph& cg
		, const dynet::Expression& i_x/*queries*/
		, const dynet::Expression& i_y/*keys and values. i_y is equal to i_x if using self_attention*/ 
		, const dynet::Expression& i_mask)
	{
		// Note: this should be done in parallel for efficiency!	
		std::vector<dynet::Expression> v_alphas(_nheads);
		for (unsigned h = 0; h < _nheads; h++){
			dynet::Expression i_Q/*queries*/ = dynet::transpose(i_x)/*Lx x num_units*/ * dynet::parameter(cg, _p_WQ[h])/*num_units x dk*/;// Lx x dk
			dynet::Expression i_K/*keys*/ = dynet::transpose(i_y)/*Ly x num_units*/ * dynet::parameter(cg, _p_WK[h])/*num_units x dk*/;// Ly x dk
			dynet::Expression i_V/*values*/ = dynet::transpose(i_y)/*Ly x num_units*/ * dynet::parameter(cg, _p_WV[h])/*num_units x dv*/;// Ly x dv

			dynet::Expression i_alpha_h;
			if (_attention_type == ATTENTION_TYPE::DOT_PRODUCT){// Luong attention type
				i_alpha_h = dynet::softmax(i_Q/*Lx x dk*/ * dynet::transpose(i_K)/*dk x Ly*/ / _att_scale) * i_V/*Ly x dv*/;// Lx x dv
			}
			else if (_attention_type == ATTENTION_TYPE::ADDITIVE_MLP){// Bahdanau attention type
				// FIXME
			}
			else assert("MultiHeadAttentionLayer: Unknown attention type!");

			v_alphas[h] = i_alpha_h;
		}

		dynet::Expression i_alpha_all = dynet::concatenate_cols(v_alphas);// Lx x (nheads * dv=num_units)
		dynet::Expression i_proj_alpha_all = dynet::transpose(i_alpha_all/*Lx x num_units*/ * dynet::parameter(cg, _p_WO)/*num_units x num_units*/);// num_units x Lx

		return i_proj_alpha_all;
	}

	// linear projection matrices
	std::vector<dynet::Parameter> _p_WQ;
	std::vector<dynet::Parameter> _p_WK;
	std::vector<dynet::Parameter> _p_WV;
	dynet::Parameter _p_WO;

	unsigned _nheads;

	// attention type
	unsigned _attention_type;

	// attention scale factor
	float _att_scale;
};
//---

//---
struct EncoderLayer{
	explicit EncoderLayer(DyNetModel* mod, const TransformerConfig& tfc)
		:_self_attention_sublayer(mod, tfc)
		, _feed_forward_sublayer(mod, tfc)
	{		
		_dropout_p = tfc._dropout_rate;

		// for layer normalisation
		_p_ln1_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln1_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
		_p_ln2_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln2_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
	}

	~EncoderLayer(){}

	float _dropout_p = 0.f;

	// multi-head attention sub-layer
	MultiHeadAttentionLayer _self_attention_sublayer;

	// position-wise feed forward sub-layer
	FeedForwardLayer _feed_forward_sublayer;

	// for layer normalisation
	dynet::Parameter _p_ln1_g, _p_ln1_b;// layer normalisation 1
	dynet::Parameter _p_ln2_g, _p_ln2_b;// layer normalisation 2

	dynet::Expression build_graph(dynet::ComputationGraph &cg, const dynet::Expression& i_src){	
		// get expressions for layer normalisation, e.g., i_ln1_g, i_ln1_b, i_ln2_g, i_ln2_b
		dynet::Expression i_ln1_g = dynet::parameter(cg, _p_ln1_g);
		dynet::Expression i_ln1_b = dynet::parameter(cg, _p_ln1_b);
		dynet::Expression i_ln2_g = dynet::parameter(cg, _p_ln2_g);
		dynet::Expression i_ln2_b = dynet::parameter(cg, _p_ln2_b);

		// create mask for self-attention
		dynet::Expression i_mask;// FIXME: how-to?

		dynet::Expression i_encl = i_src;
		
		// multi-head attention sub-layer
		dynet::Expression i_mh_att = _self_attention_sublayer.build_graph(cg, i_encl, i_encl, i_mask);

		// dropout
		i_encl = i_encl + dynet::dropout_dim(i_mh_att, 1, _dropout_p);// w/ residual connection

		// layer normalisation 1
		i_encl = layer_norm_dim(i_encl, i_ln1_g, i_ln1_b, 1);

		// point-wise feed-forward sub-layer
		dynet::Expression i_ff = _feed_forward_sublayer.build_graph(cg, i_encl);
		i_encl = i_encl + dynet::dropout_dim(i_ff, 1, _dropout_p);// w/ residual connection

		// layer normalisation 2
		i_encl = layer_norm_dim(i_encl, i_ln2_g, i_ln2_b, 1);

		return i_encl;
	}
};

struct Encoder{
	explicit Encoder(DyNetModel* mod, const TransformerConfig& tfc){
		_p_embed_s = mod->add_lookup_parameters(tfc._src_vocab_size, {tfc._num_units});

		if (tfc._position_encoding == 1){
			_p_embed_pos = mod->add_lookup_parameters(tfc._max_length, {tfc._num_units});
		}

		for (unsigned l = 0; l < tfc._nlayers; l++){
			_v_enc_layers.push_back(EncoderLayer(mod, tfc));
		}

		_sm = tfc._sm;

		_position_encoding = tfc._position_encoding;

		_dropout_p = tfc._dropout_rate;

		_scale_emb = sqrt(tfc._num_units);
	}

	~Encoder(){}

	dynet::LookupParameter _p_embed_s;// source embeddings

	dynet::LookupParameter _p_embed_pos;// position embeddings
	unsigned _position_encoding = 1;

	std::vector<EncoderLayer> _v_enc_layers;// stack of identical encoder layers

	SentinelMarkers _sm;

	float _dropout_p = 0.f;

	float _scale_emb = 0.f;

	//--- intermediate variables
	unsigned max_slen = 0;
	//---

	dynet::Expression compute_embeddings(dynet::ComputationGraph &cg, const WordIdSentences& sents/*batch of sentences*/){
		// get max length in a batch
		size_t max_len = sents[0].size();
		for(size_t i = 1; i < sents.size(); i++) max_len = std::max(max_len, sents[i].size());
		max_slen = max_len;
	
		// source encoding
		std::vector<dynet::Expression> source_embeddings;   
		std::vector<unsigned> words(sents.size());
		for (unsigned l = 0; l < max_slen; l++){
			for (unsigned bs = 0; bs < sents.size(); ++bs){
				words[bs] = (l < sents[bs].size()) ? (unsigned)sents[bs][l] : (unsigned)_sm._kSRC_EOS;
			}

			source_embeddings.push_back(lookup(cg, _p_embed_s, words));
		}

		dynet::Expression i_src = concatenate_cols(source_embeddings);// batch_size x d_model x max_slen
		i_src = i_src * _scale_emb;// scaled embeddings

		// + postional encoding
		if (_position_encoding == 1){// learned positional embedding 
			std::vector<dynet::Expression> pos_embeddings;  
			std::vector<unsigned> positions(sents.size());
			for (unsigned l = 0; l < max_slen; l++){
				for (unsigned bs = 0; bs < sents.size(); ++bs) 
					positions[bs] = l;

				pos_embeddings.push_back(lookup(cg, _p_embed_pos, positions));
			}
			dynet::Expression i_pos = concatenate_cols(pos_embeddings);// batch_size x d_model x max_slen

			i_src = i_src + i_pos;
		}
		else if (_position_encoding == 2){// sinusoidal positional encoding
			// FIXME: not yet implemented since sin and cos functions are not available yet in DyNet!
		}

		i_src = dynet::dropout(i_src, _dropout_p);// apply dropout

		return i_src;
	}

	dynet::Expression build_graph(dynet::ComputationGraph &cg, const WordIdSentences& ssents/*batch of sentences*/){
		// compute source (+ postion) embeddings
		dynet::Expression i_src = compute_embeddings(cg, ssents);
		
		dynet::Expression i_l_out = i_src;
		for (auto enc : _v_enc_layers){
			i_l_out = enc.build_graph(cg, i_l_out);// each position in the encoder can attend to all positions in the previous layer of the encoder.
			// FIXME: should we have residual connection here?
			// e.g., 
			// dynet::Expression i_l = enc.build_graph(cg, i_l_out);
			// i_l_out = i_l_out + i_l;
		}

		return i_l_out;
	}
};

typedef std::shared_ptr<Encoder> EncoderPointer;
//---

//---
struct TransformerModel {

public:
	explicit TransformerModel(const TransformerConfig& tfc);

	~TransformerModel();

protected:

	DyNetModelPointer _all_params;// all model parameters live in this object pointer. This object will be automatically released once unused!

	EncoderPointer _encoder;// encoder

	TransformerConfig _tfc;// local configuration storage
};

TransformerModel::TransformerModel(const TransformerConfig& tfc)
: _tfc(tfc)
{
	_all_params.reset(new DyNetModel());// create new model parameter object

	_encoder.reset(new Encoder(_all_params.get(), _tfc));// create new encoder object
}
//---

}; // namespace transformer

