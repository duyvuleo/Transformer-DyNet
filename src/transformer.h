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

// Utilities
#include "utils.h"

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
	float _label_smoothing_weight = 0.9f;

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
		, float label_smoothing_weight
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
		_label_smoothing_weight = label_smoothing_weight;
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
		_label_smoothing_weight = tfc._label_smoothing_weight;
		_position_encoding = tfc._position_encoding;
		_max_length = tfc._max_length;
		_sm = tfc._sm;
		_attention_type = tfc._attention_type;
	}
};
//---

// --- 
struct ModelStats {
	double _loss = 0.0f;
	unsigned _words_src = 0;
	unsigned _words_tgt = 0;
	unsigned _words_src_unk = 0;
	unsigned _words_tgt_unk = 0;

	ModelStats(){}
};
// --- 

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
		: _innerConv(mod, tfc._num_units, tfc._num_units * 4/*4 according to the paper*/)
		, _outerConv(mod, tfc._num_units * 4, tfc._num_units)
	{		
	}	

	~FeedForwardLayer(){}	

	dynet::Expression build_graph(dynet::ComputationGraph& cg, const dynet::Expression& i_inp/*num_units x L*/){
		// FFN(x) = relu(x * W1 + b1) * W2 + b2
		dynet::Expression i_inner = dynet::reshape(i_inp, {1, i_inp.dim().d[1], i_inp.dim().d[0]});
		i_inner = _innerConv.apply(cg, i_inner);// x * W1 + b1
		i_inner = dynet::rectify/*swish*/(i_inner);// relu or swish?
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
	explicit MultiHeadAttentionLayer(DyNetModel* mod, const TransformerConfig& tfc, bool is_future_blinding = false)
	{
		_p_WQ.resize(tfc._nheads);
		_p_WK.resize(tfc._nheads);
		_p_WV.resize(tfc._nheads);
		for (unsigned h = 0; h < tfc._nheads; h++){
			_p_WQ[h] = mod->add_parameters({tfc._num_units / tfc._nheads, tfc._num_units});// dk = tfc._num_units / tfc._nheads
			_p_WK[h] = mod->add_parameters({tfc._num_units / tfc._nheads, tfc._num_units});// dk
			_p_WV[h] = mod->add_parameters({tfc._num_units / tfc._nheads, tfc._num_units});// dv = tfc._num_units / tfc._nheads
		}

		_p_WO = mod->add_parameters({tfc._num_units, tfc._num_units});
		
		_att_scale = sqrt(tfc._num_units / tfc._nheads);

		_attention_type = tfc._attention_type;

		_nheads = tfc._nheads;

		_is_future_blinding = is_future_blinding;

		_dropout_p = tfc._dropout_rate;
	}

	~MultiHeadAttentionLayer(){}

	// linear projection matrices
	std::vector<dynet::Parameter> _p_WQ;
	std::vector<dynet::Parameter> _p_WK;
	std::vector<dynet::Parameter> _p_WV;
	dynet::Parameter _p_WO;

	unsigned _nheads = 0;// number of heads

	// attention type
	unsigned _attention_type= ATTENTION_TYPE::DOT_PRODUCT;

	// attention scale factor
	float _att_scale = 0.f;

	bool _is_future_blinding = false;

	float _dropout_p = 0.f;

	dynet::Expression build_graph(dynet::ComputationGraph& cg
		, const dynet::Expression& i_x/*queries*/
		, const dynet::Expression& i_y/*keys and values. i_y is equal to i_x if using self_attention*/)
	{
		// create mask for self-attention in decoder
		dynet::Dim dim = i_x.dim();
		dynet::Expression i_mask = dynet::zeros(cg, dim);// zero masking if not for future blinding
		if (_is_future_blinding) i_mask = create_triangle_mask(cg, dim[1]/*Lx*/, false);
		
		// Note: this should be done in parallel for efficiency!
		// e.g., utilising pseudo-batching?	
		std::vector<dynet::Expression> v_atts(_nheads);
		for (unsigned h = 0; h < _nheads; h++){
			dynet::Expression i_Q/*queries*/ = dynet::parameter(cg, _p_WQ[h])/*dk x num_units*/ * i_x/*num_units x Lx*/;// dk x Lx
			dynet::Expression i_K/*keys*/ = dynet::parameter(cg, _p_WK[h])/*dk x num_units*/ * i_y/*num_units x Ly*/;// dk x Ly
			dynet::Expression i_V/*values*/ = dynet::parameter(cg, _p_WK[h])/*dv x num_units*/ * i_y/*num_units x Ly*/;// dv x Ly

			dynet::Expression i_att_h;
			if (_attention_type == ATTENTION_TYPE::DOT_PRODUCT){// Luong attention type
				dynet::Expression i_alpha_pre = (dynet::transpose(i_K)/*Ly * dk*/ * i_Q/*dk x Lx*/) / _att_scale;// Ly x Lx (unnormalised) 

				dynet::Expression i_alpha = dynet::softmax(i_alpha_pre + i_mask);// Ly x Lx (normalised, col-major)
				// FIXME: save the soft alignment in i_alpha if necessary!
				
				// attention dropout (col-major or whole matrix?)
				//i_alpha = dynet::dropout_dim(i_alpha, 0/*col-major*/, _dropout_p);
				i_alpha = dynet::dropout(i_alpha, _dropout_p);// for whole matrix

				i_att_h = i_V/*dv x Ly*/ * i_alpha/*Ly x Lx*/;// dv x Lx
			}
			else if (_attention_type == ATTENTION_TYPE::ADDITIVE_MLP){// Bahdanau attention type
				// FIXME
			}
			else assert("MultiHeadAttentionLayer: Unknown attention type!");

			v_atts[h] = i_att_h;
		}

		// joint all head attentions
		dynet::Expression i_atts = dynet::concatenate(v_atts);// (nheads * dv=num_units) x Lx

		// linear projection
		dynet::Expression i_proj_atts = dynet::parameter(cg, _p_WO)/*num_units x num_units*/ * i_atts/*num_units x Lx*/;// num_units x Lx

		return i_proj_atts;
	}
};
//---

//---
struct EncoderLayer{
	explicit EncoderLayer(DyNetModel* mod, const TransformerConfig& tfc)
		:_self_attention_sublayer(mod, tfc)
		, _feed_forward_sublayer(mod, tfc)
	{		
		// for layer normalisation
		_p_ln1_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln1_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
		_p_ln2_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln2_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
	}

	~EncoderLayer(){}

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

		dynet::Expression i_encl = i_src;
		
		// multi-head attention sub-layer
		dynet::Expression i_mh_att = _self_attention_sublayer.build_graph(cg, i_encl, i_encl);

		// w/ residual connection
		i_encl = i_encl + i_mh_att;

		// layer normalisation 1
		i_encl = layer_norm_matrix(i_encl, i_ln1_g, i_ln1_b);

		// position-wise feed-forward sub-layer
		dynet::Expression i_ff = _feed_forward_sublayer.build_graph(cg, i_encl);

		// w/ residual connection
		i_encl = i_encl + i_ff;

		// layer normalisation 2
		i_encl = layer_norm_matrix(i_encl, i_ln2_g, i_ln2_b);

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

	unsigned _batch_slen = 0;

	dynet::Expression compute_embeddings(dynet::ComputationGraph &cg, const WordIdSentences& sents/*batch of sentences*/){
		// get max length in a batch
		size_t max_len = sents[0].size();
		for(size_t i = 1; i < sents.size(); i++) max_len = std::max(max_len, sents[i].size());
		_batch_slen = max_len;
	
		// source encoding
		std::vector<dynet::Expression> source_embeddings;   
		std::vector<unsigned> words(sents.size());
		for (unsigned l = 0; l < max_len; l++){
			for (unsigned bs = 0; bs < sents.size(); ++bs){
				words[bs] = (l < sents[bs].size()) ? (unsigned)sents[bs][l] : (unsigned)_sm._kSRC_EOS;
			}

			source_embeddings.push_back(lookup(cg, _p_embed_s, words));
		}

		dynet::Expression i_src = concatenate_cols(source_embeddings);// (batch_size x) num_units x _batch_slen
		i_src = i_src * _scale_emb;// scaled embeddings

		// + postional encoding
		if (_position_encoding == 1){// learned positional embedding 
			std::vector<dynet::Expression> pos_embeddings;  
			std::vector<unsigned> positions(sents.size());
			for (unsigned l = 0; l < max_len; l++){
				for (unsigned bs = 0; bs < sents.size(); ++bs) 
					positions[bs] = l;

				pos_embeddings.push_back(lookup(cg, _p_embed_pos, positions));
			}
			dynet::Expression i_pos = concatenate_cols(pos_embeddings);// (batch_size x) num_units x _batch_slen

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
			// stacking approach
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
struct DecoderLayer{
	explicit DecoderLayer(DyNetModel* mod, const TransformerConfig& tfc)
		:_self_attention_sublayer(mod, tfc, true)
		, _src_attention_sublayer(mod, tfc)
		, _feed_forward_sublayer(mod, tfc)
	{	
		// initialisation for layer normalisation
		_p_ln1_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln1_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
		_p_ln2_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln2_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
		_p_ln3_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln3_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
	}

	~DecoderLayer(){}	

	// multi-head attention sub-layers
	MultiHeadAttentionLayer _self_attention_sublayer;// self-attention
	MultiHeadAttentionLayer _src_attention_sublayer;// source attention

	// position-wise feed forward sub-layer
	FeedForwardLayer _feed_forward_sublayer;

	// for layer normalisation
	dynet::Parameter _p_ln1_g, _p_ln1_b;// layer normalisation 1
	dynet::Parameter _p_ln2_g, _p_ln2_b;// layer normalisation 2
	dynet::Parameter _p_ln3_g, _p_ln3_b;// layer normalisation 3

	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const dynet::Expression& i_enc_inp
		, const dynet::Expression& i_dec_inp)
	{	
		// get expressions for layer normalisation, e.g., i_ln1_g, i_ln1_b, i_ln2_g, i_ln2_b, i_ln3_g, i_ln3_b
		dynet::Expression i_ln1_g = dynet::parameter(cg, _p_ln1_g);
		dynet::Expression i_ln1_b = dynet::parameter(cg, _p_ln1_b);
		dynet::Expression i_ln2_g = dynet::parameter(cg, _p_ln2_g);
		dynet::Expression i_ln2_b = dynet::parameter(cg, _p_ln2_b);
		dynet::Expression i_ln3_g = dynet::parameter(cg, _p_ln3_g);
		dynet::Expression i_ln3_b = dynet::parameter(cg, _p_ln3_b);
	
		dynet::Expression i_decl = i_dec_inp;
		
		// multi-head self attention sub-layer
		dynet::Expression i_mh_self_att = _self_attention_sublayer.build_graph(cg, i_decl, i_decl);

		// w/ residual connection
		i_decl = i_decl + i_mh_self_att;

		// layer normalisation 1
		i_decl = layer_norm_matrix(i_decl, i_ln1_g, i_ln1_b);

		// multi-head source attention sub-layer
		dynet::Expression i_mh_src_att = _src_attention_sublayer.build_graph(cg, i_decl, i_enc_inp);

		// w/ residual connection
		i_decl = i_decl + i_mh_src_att;

		// layer normalisation 2
		i_decl = layer_norm_matrix(i_decl, i_ln2_g, i_ln2_b);// position-wise

		// position-wise feed-forward sub-layer
		dynet::Expression i_ff = _feed_forward_sublayer.build_graph(cg, i_decl);

		// w/ residual connection
		i_decl = i_decl + i_ff;

		// layer normalisation 3
		i_decl = layer_norm_matrix(i_decl, i_ln3_g, i_ln3_b);// position-wise

		return i_decl;
	}
};

struct Decoder{
	explicit Decoder(DyNetModel* mod, const TransformerConfig& tfc){
		_p_embed_t = mod->add_lookup_parameters(tfc._tgt_vocab_size, {tfc._num_units});

		if (tfc._position_encoding == 1){
			_p_embed_pos = mod->add_lookup_parameters(tfc._max_length, {tfc._num_units});
		}

		for (unsigned l = 0; l < tfc._nlayers; l++){
			_v_dec_layers.push_back(DecoderLayer(mod, tfc));
		}		

		_sm = tfc._sm;

		_position_encoding = tfc._position_encoding;

		_dropout_p = tfc._dropout_rate;

		_scale_emb = sqrt(tfc._num_units);
	}

	~Decoder(){}

	dynet::LookupParameter _p_embed_t;// source embeddings

	dynet::LookupParameter _p_embed_pos;// position embeddings
	unsigned _position_encoding = 1;

	std::vector<DecoderLayer> _v_dec_layers;// stack of identical decoder layers

	SentinelMarkers _sm;

	float _dropout_p = 0.f;

	float _scale_emb = 0.f;

	// --- intermediate variables
	unsigned _batch_tlen = 0;
	// ---

	dynet::Expression get_wrd_embedding_matrix(dynet::ComputationGraph &cg){
		return dynet::parameter(cg, _p_embed_t);// target word embedding matrix (num_units x |V_T|)
	}

	dynet::Expression compute_embeddings(dynet::ComputationGraph &cg, const WordIdSentences& sents/*batch of sentences*/){
		// get max length in a batch
		size_t max_len = sents[0].size();
		for(size_t i = 1; i < sents.size(); i++) max_len = std::max(max_len, sents[i].size());
		_batch_tlen = max_len;
	
		// source encoding
		std::vector<dynet::Expression> target_embeddings;   
		std::vector<unsigned> words(sents.size());
		for (unsigned l = 0; l < max_len; l++){
			for (unsigned bs = 0; bs < sents.size(); ++bs){
				words[bs] = (l < sents[bs].size()) ? (unsigned)sents[bs][l] : (unsigned)_sm._kTGT_EOS;
			}

			target_embeddings.push_back(lookup(cg, _p_embed_t, words));
		}

		dynet::Expression i_tgt = concatenate_cols(target_embeddings);// (batch_size x) num_units x (_batch_tlen-1)
		i_tgt = i_tgt * _scale_emb;// scaled embeddings

		// + postional encoding
		if (_position_encoding == 1){// learned positional embedding 
			std::vector<dynet::Expression> pos_embeddings;  
			std::vector<unsigned> positions(sents.size());
			for (unsigned l = 0; l < max_len; l++){
				for (unsigned bs = 0; bs < sents.size(); ++bs) 
					positions[bs] = l;

				pos_embeddings.push_back(lookup(cg, _p_embed_pos, positions));
			}
			dynet::Expression i_pos = concatenate_cols(pos_embeddings);// (batch_size x) num_units x (_batch_tlen-1)

			i_tgt = i_tgt + i_pos;
		}
		else if (_position_encoding == 2){// sinusoidal positional encoding
			// FIXME: not yet implemented since sin and cos functions are not available yet in DyNet!
		}

		i_tgt = dynet::dropout(i_tgt, _dropout_p);// apply dropout

		return i_tgt;
	}

	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const WordIdSentences& tsents/*batch of sentences*/
		, const dynet::Expression& i_src)
	{
		// compute source (+ postion) embeddings
		dynet::Expression i_tgt = compute_embeddings(cg, tsents);
		
		dynet::Expression i_l_out = i_tgt;
		for (auto dec : _v_dec_layers){
			// stacking approach
			i_l_out = dec.build_graph(cg, i_src, i_l_out);// each position in the decoder can attend to all positions (up to and including the current position) in the previous layer of the decoder.
			// FIXME: should we have residual connection here?
			// e.g., 
			// dynet::Expression i_l = dec.build_graph(cg, i_src, i_l_out);
			// i_l_out = i_l_out + i_l;
		}

		return i_l_out;// (batch_size x) num_units x (Ly-1)
	}
};
typedef std::shared_ptr<Decoder> DecoderPointer;
//---

//---
struct TransformerModel {

public:
	explicit TransformerModel(const TransformerConfig& tfc, dynet::Dict& sd, dynet::Dict& td);

	~TransformerModel(){}

	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const WordIdSentences& ssents/*batched*/
		, const WordIdSentences& tsents/*batched*/
		, const ModelStats &stats = ModelStats());

	dynet::ParameterCollection& get_model_parameters();
	void initialise_params_from_file(const string &params_file);
	void save_params_to_file(const string &params_file);

	void set_dropout(bool is_activated = true);

protected:

	DyNetModelPointer _all_params;// all model parameters live in this object pointer. This object will be automatically released once unused!

	EncoderPointer _encoder;// encoder

	DecoderPointer _decoder;// decoder

	std::pair<dynet::Dict, dynet::Dict> _dicts;// pair of source and target vocabularies

	dynet::Parameter _p_Wo_bias;// bias of final linear projection layer

	TransformerConfig _tfc;// local configuration storage
};

TransformerModel::TransformerModel(const TransformerConfig& tfc, dynet::Dict& sd, dynet::Dict& td)
: _tfc(tfc)
{
	_all_params.reset(new DyNetModel());// create new model parameter object

	_encoder.reset(new Encoder(_all_params.get(), _tfc));// create new encoder object

	_decoder.reset(new Decoder(_all_params.get(), _tfc));// create new decoder object

	// final output projection layer
	_p_Wo_bias = _all_params.get()->add_parameters({tfc._tgt_vocab_size});// optional

	// dictionaries
	_dicts.first = sd;
	_dicts.second = td;
}

dynet::Expression TransformerModel::build_graph(dynet::ComputationGraph &cg
	, const WordIdSentences& ssents
	, const WordIdSentences& tsents
	, const ModelStats &stats)
{
	// encode source
	dynet::Expression i_src = _encoder.get()->build_graph(cg, ssents);// (batch_size x) num_units x Lx

	// decode target
	dynet::Expression i_tgt = _decoder.get()->build_graph(cg, tsents, i_src);// (batch_size x) num_units x (Ly - 1)

	// get losses	
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix)

	std::vector<dynet::Expression> v_errors;
	unsigned tlen = _decoder.get()->_batch_tlen;
	std::vector<unsigned> next_words(tsents.size());
	for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
		for(size_t bs = 0; bs < tsents.size(); bs++)
			next_words[bs] = (tsents[bs].size() > (t+1)) ? (unsigned)tsents[bs][t + 1] : _tfc._sm._kTGT_EOS;
	
		// output linear projections (logit)
		dynet::Expression i_err;
		if (!_tfc._use_label_smoothing){
			dynet::Expression i_tgt_t = dynet::select_cols(i_tgt, {t + 1});// shifted right
			dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)
		
			// log_softmax and loss
			i_err = dynet::pickneglogsoftmax(i_r_t, next_words);		
		}
		else{// w/ label smoothing
			// FIXME
		}

		v_errors.push_back(i_err);
	}

	dynet::Expression i_tloss = dynet::sum_batches(dynet::sum(v_errors));

	return i_tloss;
}

dynet::ParameterCollection& TransformerModel::get_model_parameters(){
	return *_all_params.get();
}

void TransformerModel::initialise_params_from_file(const string &params_file)
{
	dynet::load_dynet_model(params_file, _all_params.get());// FIXME: use binary streaming instead for saving disk spaces?
}

void TransformerModel::save_params_to_file(const string &params_file)
{
	dynet::save_dynet_model(params_file, _all_params.get());
}

void TransformerModel::set_dropout(bool is_activated){
	// FIXME:
}

//---

}; // namespace transformer



