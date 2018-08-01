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

// All utilities
#include "utils.h"

// Layers
#include "layers.h"

using namespace std;
using namespace dynet;

namespace transformer {

//--- LM Decoder Layer
struct LMDecoderLayer{
	explicit LMDecoderLayer(DyNetModel* mod, TransformerConfig& tfc)
		:_self_attention_sublayer(mod, tfc, true)
		, _feed_forward_sublayer(mod, tfc)
	{	
		// initialisation for layer normalisation
		_p_ln1_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln1_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
		_p_ln2_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln2_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));

		_p_tfc = &tfc;
	}

	~LMDecoderLayer(){}	

	// multi-head attention sub-layers
	MultiHeadAttentionLayer _self_attention_sublayer;// self-attention

	// position-wise feed forward sub-layer
	FeedForwardLayer _feed_forward_sublayer;

	// for layer normalisation
	dynet::Parameter _p_ln1_g, _p_ln1_b;// layer normalisation 1
	dynet::Parameter _p_ln2_g, _p_ln2_b;// layer normalisation 2

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;

	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const dynet::Expression& i_dec_inp
		, const MaskBase& self_mask)
	{	
		// get expressions for layer normalisation, e.g., i_ln1_g, i_ln1_b, i_ln2_g, i_ln2_b, i_ln3_g, i_ln3_b
		dynet::Expression i_ln1_g = dynet::parameter(cg, _p_ln1_g);
		dynet::Expression i_ln1_b = dynet::parameter(cg, _p_ln1_b);
		dynet::Expression i_ln2_g = dynet::parameter(cg, _p_ln2_g);
		dynet::Expression i_ln2_b = dynet::parameter(cg, _p_ln2_b);
	
		dynet::Expression i_decl = i_dec_inp;
		
		// multi-head self attention sub-layer
		dynet::Expression i_mh_self_att = _self_attention_sublayer.build_graph(cg, i_decl, i_decl, self_mask);// ((num_units, Ly), batch_size)

		// dropout to the output of sub-layer
		if (_p_tfc->_use_dropout && _p_tfc->_decoder_sublayer_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_mh_self_att = dynet::dropout_dim(i_mh_self_att, 1/*col-major*/, _p_tfc->_decoder_sublayer_dropout_rate);// col-wise dropout
#else
			i_mh_self_att = dynet::dropout(i_mh_self_att, _p_tfc->_decoder_sublayer_dropout_rate);// full dropout
#endif

		// w/ residual connection
		i_decl = i_decl + i_mh_self_att;// ((num_units, Ly), batch_size)

		// layer normalisation 1
		i_decl = layer_norm_colwise_3(i_decl, i_ln1_g, i_ln1_b);// ((num_units, Ly), batch_size)

		// position-wise feed-forward sub-layer
		dynet::Expression i_ff = _feed_forward_sublayer.build_graph(cg, i_decl);// ((num_units, Ly), batch_size)

		// w/ residual connection
		i_decl = i_decl + i_ff;

		// layer normalisation 3
		i_decl = layer_norm_colwise_3(i_decl, i_ln2_g, i_ln2_b);// ((num_units, Ly), batch_size)

		return i_decl;
	}
};

struct LMDecoder{
	explicit LMDecoder(DyNetModel* mod, TransformerConfig& tfc)
	{
		_p_embed_t = mod->add_lookup_parameters(tfc._tgt_vocab_size, {tfc._num_units});

		if (!tfc._use_hybrid_model && tfc._position_encoding == 1){
			_p_embed_pos = mod->add_lookup_parameters(tfc._max_length, {tfc._num_units});
		}

		for (unsigned l = 0; l < tfc._nlayers; l++){
			_v_dec_layers.push_back(LMDecoderLayer(mod, tfc));
		}

		if (tfc._use_hybrid_model){
			_p_tgt_rnn.reset(new dynet::LSTMBuilder(tfc._nlayers, tfc._num_units, tfc._num_units, *mod, true/*w/ layer norm*/));
		}

		_scale_emb = std::sqrt(tfc._num_units);

		_p_tfc = &tfc;
	}

	~LMDecoder(){}

	dynet::LookupParameter _p_embed_t;// source embeddings
	dynet::LookupParameter _p_embed_pos;// position embeddings

	// hybrid architecture: use LSTM-based RNN over word embeddings instead of word embeddings + positional encodings
	std::shared_ptr<dynet::LSTMBuilder> _p_tgt_rnn;

	std::vector<LMDecoderLayer> _v_dec_layers;// stack of identical decoder layers

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;

	// --- intermediate variables
	float _scale_emb = 0.f;
	unsigned _batch_tlen = 0;
	// decoder masks
	MaskBase _self_mask;
	// ---

	dynet::Expression get_wrd_embedding_matrix(dynet::ComputationGraph &cg){
		return dynet::parameter(cg, _p_embed_t);// target word embedding matrix (num_units x |V_T|)
	}

	dynet::Expression compute_embeddings_and_masks(dynet::ComputationGraph &cg
		, const WordIdSentences& sents/*batch of target sentences*/)
	{
		// compute embeddings			
		// get max length in a batch
		size_t max_len = sents[0].size();
		for(size_t i = 1; i < sents.size(); i++) max_len = std::max(max_len, sents[i].size());
		_batch_tlen = max_len;

		std::vector<dynet::Expression> target_embeddings;   
		std::vector<unsigned> words(sents.size());
		std::vector<std::vector<float>> v_seq_masks(sents.size());
		dynet::Expression i_tgt;
		if (_p_tfc->_use_hybrid_model){
			// target embeddings via RNN
			_p_tgt_rnn->new_graph(cg);
			_p_tgt_rnn->start_new_sequence();
			for (unsigned l = 0; l < max_len - (_p_tfc->_is_training)?1:0; l++){// offset by 1 during training
				for (unsigned bs = 0; bs < sents.size(); ++bs)
				{
					//words[bs] = (l < sents[bs].size()) ? (unsigned)sents[bs][l] : (unsigned)_p_tfc->_sm._kTGT_EOS;
					if (l < sents[bs].size()){
						words[bs] = (unsigned)sents[bs][l];
						v_seq_masks[bs].push_back(0.f);// padding position
					}
					else{
						words[bs] = (unsigned)_p_tfc->_sm._kTGT_EOS;
						v_seq_masks[bs].push_back(1.f);// padding position
					}
				}

				target_embeddings.push_back(_p_tgt_rnn->add_input(dynet::lookup(cg, _p_embed_t, words)));
			}
			i_tgt = dynet::concatenate_cols(target_embeddings);// ((num_units, Ly), batch_size)
		}
		else{
			// target embeddings
			for (unsigned l = 0; l < max_len - (_p_tfc->_is_training)?1:0; l++){// offset by 1 during training
				for (unsigned bs = 0; bs < sents.size(); ++bs)
				{
					//words[bs] = (l < sents[bs].size()) ? (unsigned)sents[bs][l] : (unsigned)_p_tfc->_sm._kTGT_EOS;
					if (l < sents[bs].size()){
						words[bs] = (unsigned)sents[bs][l];
						v_seq_masks[bs].push_back(0.f);// padding position
					}
					else{
						words[bs] = (unsigned)_p_tfc->_sm._kTGT_EOS;
						v_seq_masks[bs].push_back(1.f);// padding position
					}
				}

				target_embeddings.push_back(dynet::lookup(cg, _p_embed_t, words));
			}
			i_tgt = dynet::concatenate_cols(target_embeddings);// ((num_units, Ly), batch_size)

			// scale
			i_tgt = i_tgt * _scale_emb;// scaled embeddings

			// + postional encoding			
			if (_p_tfc->_position_encoding == 1){// learned positional embedding 
				std::vector<dynet::Expression> pos_embeddings;  
				std::vector<unsigned> positions(sents.size());
				for (unsigned l = 0; l < max_len - (_p_tfc->_is_training)?1:0; l++){// offset by 1 during training
					for (unsigned bs = 0; bs < sents.size(); ++bs){
						if (l >= _p_tfc->_max_length) positions[bs] = _p_tfc->_max_length - 1;// Trick: if using learned position encoding, during decoding/inference, sentence length may be longer than fixed max length. We overcome this by tying to _p_tfc._max_length.
						else
							positions[bs] = l;
				}

					pos_embeddings.push_back(dynet::lookup(cg, _p_embed_pos, positions));
				}
				dynet::Expression i_pos = dynet::concatenate_cols(pos_embeddings);// // ((num_units, Ly), batch_size)

				i_tgt = i_tgt + i_pos;
			}
			else if (_p_tfc->_position_encoding == 2){// sinusoidal positional encoding
				dynet::Expression i_pos = make_sinusoidal_position_encoding(cg, i_tgt.dim());

				i_tgt = i_tgt + i_pos;
			}
			else if (_p_tfc->_position_encoding != 0) TRANSFORMER_RUNTIME_ASSERT("Unknown positional encoding type!");
		}

		// dropout to the sums of the embeddings and the positional encodings
		if (_p_tfc->_use_dropout && _p_tfc->_decoder_emb_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_tgt = dynet::dropout_dim(i_tgt, 1/*col-major*/, _p_tfc->_decoder_emb_dropout_rate);// col-wise dropout
#else
			i_tgt = dynet::dropout(i_tgt, _p_tfc->_decoder_emb_dropout_rate);// full dropout		
#endif

		// create maskings
		// self-attention
		// for future blinding
		_self_mask.create_future_blinding_mask(cg, i_tgt.dim()[1]);

		// for padding positions blinding
		_self_mask.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
		_self_mask.create_padding_positions_masks(_p_tfc->_nheads);
#else 
		_self_mask.create_padding_positions_masks(1);
#endif

		return i_tgt;
	}

	dynet::Expression compute_embeddings_and_masks(dynet::ComputationGraph &cg
		, const std::vector<dynet::Expression>& v_soft_targets/*batch of soft sentences*/)
	{
		// compute embeddings			
		// get max length in a batch
		// get max length in a batch
		unsigned bsize = v_soft_targets[0].dim().batch_elems();
		_batch_tlen = v_soft_targets.size();

		dynet::Expression i_tgt;
		if (_p_tfc->_use_hybrid_model){
			// target embeddings via RNN
			std::vector<dynet::Expression> target_embeddings; 
			_p_tgt_rnn->new_graph(cg);
			_p_tgt_rnn->start_new_sequence();
			for (unsigned l = 0; l < _batch_tlen - (_p_tfc->_is_training)?1:0; l++){// offset by 1 during training
				target_embeddings.push_back(_p_tgt_rnn->add_input(v_soft_targets[l]));
			}

			i_tgt = dynet::concatenate_cols(target_embeddings);// ((num_units, Ly), batch_size)
		}
		else{
			// target embeddings
			i_tgt = dynet::concatenate_cols(v_soft_targets);// ((num_units, Ly), batch_size)

			// scale
			i_tgt = i_tgt * _scale_emb;// scaled embeddings

			// + postional encoding			
			if (_p_tfc->_position_encoding == 1){// learned positional embedding 
				std::vector<dynet::Expression> pos_embeddings;  
				std::vector<unsigned> positions(bsize);
				for (unsigned l = 0; l < _batch_tlen - (_p_tfc->_is_training)?1:0; l++){// offset by 1 during training
					for (unsigned bs = 0; bs < bsize; ++bs){
						if (l >= _p_tfc->_max_length) positions[bs] = _p_tfc->_max_length - 1;// Trick: if using learned position encoding, during decoding/inference, sentence length may be longer than fixed max length. We overcome this by tying to _p_tfc._max_length.
						else
							positions[bs] = l;
				}

					pos_embeddings.push_back(dynet::lookup(cg, _p_embed_pos, positions));
				}
				dynet::Expression i_pos = dynet::concatenate_cols(pos_embeddings);// // ((num_units, Ly), batch_size)

				i_tgt = i_tgt + i_pos;
			}
			else if (_p_tfc->_position_encoding == 2){// sinusoidal positional encoding
				dynet::Expression i_pos = make_sinusoidal_position_encoding(cg, i_tgt.dim());

				i_tgt = i_tgt + i_pos;
			}
			else if (_p_tfc->_position_encoding != 0) TRANSFORMER_RUNTIME_ASSERT("Unknown positional encoding type!");
		}

		// dropout to the sums of the embeddings and the positional encodings
		if (_p_tfc->_use_dropout && _p_tfc->_decoder_emb_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_tgt = dynet::dropout_dim(i_tgt, 1/*col-major*/, _p_tfc->_decoder_emb_dropout_rate);// col-wise dropout
#else
			i_tgt = dynet::dropout(i_tgt, _p_tfc->_decoder_emb_dropout_rate);// full dropout		
#endif

		// create maskings
		std::vector<std::vector<float>> v_seq_masks(bsize, std::vector<float>(_batch_tlen, 0.f));
		// self-attention
		// for future blinding
		_self_mask.create_future_blinding_mask(cg, i_tgt.dim()[1]);

		// for padding positions blinding
		_self_mask.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
		_self_mask.create_padding_positions_masks(_p_tfc->_nheads);
#else 
		_self_mask.create_padding_positions_masks(1);
#endif

		return i_tgt;
	}

	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const WordIdSentences& tsents/*batch of sentences*/)
	{		
		// compute target (+ postion) embeddings
		dynet::Expression i_tgt_rep = compute_embeddings_and_masks(cg, tsents);// ((num_units, Ly), batch_size)
			
		dynet::Expression i_dec_l_out = i_tgt_rep;
		for (auto dec : _v_dec_layers){
			// stacking approach
			i_dec_l_out = dec.build_graph(cg, i_dec_l_out, _self_mask);// each position in the decoder can attend to all positions (up to and including the current position) in the previous layer of the decoder.
		}
	
		return i_dec_l_out;// ((num_units, Ly), batch_size)
	}

	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const std::vector<dynet::Expression>& v_soft_tsents/*batch of soft sentences*/)
	{		
		// compute target (+ postion) embeddings
		dynet::Expression i_tgt_rep = compute_embeddings_and_masks(cg, v_soft_tsents);// ((num_units, Ly), batch_size)
			
		dynet::Expression i_dec_l_out = i_tgt_rep;
		for (auto dec : _v_dec_layers){
			// stacking approach
			i_dec_l_out = dec.build_graph(cg, i_dec_l_out, _self_mask);// each position in the decoder can attend to all positions (up to and including the current position) in the previous layer of the decoder.
		}
	
		return i_dec_l_out;// ((num_units, Ly), batch_size)
	}

};
typedef std::shared_ptr<LMDecoder> LMDecoderPointer;
//---

//--- Transformer Language Model
struct TransformerLModel {

public:
	explicit TransformerLModel(const TransformerConfig& tfc, dynet::Dict& d);

	explicit TransformerLModel();

	~TransformerLModel(){}

	// for training
	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const WordIdSentences& sents/*batched*/
		, ModelStats* pstats=nullptr
		, bool is_eval_on_dev=false);	
	void get_avg_losses(dynet::ComputationGraph &cg
		, const WordIdSentences& tsents
		, std::vector<float>& v_losses
		, bool do_sum=false);
	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const std::vector<dynet::Expression>& v_soft_sents/*batched*/
		, bool is_eval_on_dev=false);
	dynet::Expression step_forward(dynet::ComputationGraph &cg
		, const WordIdSentences &partial_sents/*batched*/
		, bool log_prob
		, std::vector<dynet::Expression> &aligns);
	void sample(dynet::ComputationGraph& cg, WordIdSentence &sampled_sent, const std::string &prefix=""/*e.g., <s>*/);// sampling

	dynet::ParameterCollection& get_model_parameters();
	void initialise_params_from_file(const std::string &params_file);
	void save_params_to_file(const std::string &params_file);

	void set_dropout(bool is_activated = true);

	dynet::Dict& get_dict();

	TransformerConfig& get_config();

protected:

	DyNetModelPointer _all_params;// all model parameters live in this object pointer. This object will be automatically released once unused!

	LMDecoderPointer _decoder;// lm decoder

	dynet::Dict _dict;// vocabulary

	dynet::Parameter _p_Wo_bias;// bias of final linear projection layer

	TransformerConfig _tfc;// local configuration storage
};

TransformerLModel::TransformerLModel(){
	_all_params = nullptr;
	_decoder = nullptr;
}

TransformerLModel::TransformerLModel(const TransformerConfig& tfc, dynet::Dict& d)
: _tfc(tfc)
{
	_all_params.reset(new DyNetModel());// create new model parameter object

	_decoder.reset(new LMDecoder(_all_params.get(), _tfc));// create new decoder object

	// final output projection layer
	_p_Wo_bias = _all_params.get()->add_parameters({_tfc._tgt_vocab_size});// optional

	// dictionaries
	_dict = d;
}

dynet::Expression TransformerLModel::step_forward(dynet::ComputationGraph &cg
	, const WordIdSentences &partial_sents
	, bool log_prob
	, std::vector<dynet::Expression> &aligns)
{
	// decode target
	// IMPROVEMENT: during decoding, some parts in partial_sent will be recomputed. This is wasteful, especially for beam search decoding.
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, partial_sents);
	dynet::Expression i_tgt_t;
	if (partial_sents[0].size() == 1) i_tgt_t = i_tgt_ctx;
	else 
		//i_tgt_t = dynet::select_cols(i_tgt_ctx, {(unsigned)(partial_sents.size() - 1)});
		i_tgt_t = dynet::pick(i_tgt_ctx, (unsigned)(partial_sents[0].size() - 1), 1);// shifted right, ((|V_T|, 1), batch_size)

	// output linear projections (w/ bias)
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859
	dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)

	// FIXME: get the alignments for visualisation

	// compute softmax prediction
	if (log_prob)
		return dynet::log_softmax(i_r_t);
	else
		return dynet::softmax(i_r_t);
}

dynet::Expression TransformerLModel::build_graph(dynet::ComputationGraph &cg
	, const WordIdSentences& tsents
	, ModelStats* pstats
	, bool is_eval_on_dev)
{	
	// decode target
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, tsents);// ((num_units, Ly), batch_size)

	// get losses	
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859

	// compute the logit and linear projections
	dynet::Expression i_r = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_ctx});// ((|V_T|, (Ly-1)), batch_size)

	std::vector<dynet::Expression> v_errors;
	unsigned tlen = _decoder.get()->_batch_tlen;
	std::vector<unsigned> next_words(tsents.size());
	for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
		for(size_t bs = 0; bs < tsents.size(); bs++){
			next_words[bs] = (tsents[bs].size() > (t + 1)) ? (unsigned)tsents[bs][t + 1] : _tfc._sm._kTGT_EOS;
			if (tsents[bs].size() > t && pstats) {
				pstats->_words_tgt++;
				if (tsents[bs][t] == _tfc._sm._kTGT_UNK) pstats->_words_tgt_unk++;
			}
		}

		// get the prediction at timestep t
		//dynet::Expression i_r_t = dynet::select_cols(i_r, {t});// shifted right, ((|V_T|, 1), batch_size)
		dynet::Expression i_r_t = dynet::pick(i_r, t, 1);// shifted right, ((|V_T|, 1), batch_size)
	
		// log_softmax and loss
		dynet::Expression i_err;
		if (_tfc._use_label_smoothing && !is_eval_on_dev/*only applies in training*/)
		{// w/ label smoothing (according to section 7.5.1 of http://www.deeplearningbook.org/contents/regularization.html) and https://arxiv.org/pdf/1512.00567v1.pdf.
			// label smoothing regularizes a model based on a softmax with k output values by replacing the hard 0 and 1 classification targets with targets of \epsilon / (k−1) and 1 − \epsilon, respectively!
			dynet::Expression i_log_softmax = dynet::log_softmax(i_r_t);
			dynet::Expression i_pre_loss = -dynet::pick(i_log_softmax, next_words);
			dynet::Expression i_ls_loss = -dynet::sum_elems(i_log_softmax) / (_tfc._tgt_vocab_size - 1);// or -dynet::mean_elems(i_log_softmax)
			i_err = (1.f - _tfc._label_smoothing_weight) * i_pre_loss + _tfc._label_smoothing_weight * i_ls_loss;
		}
		else 
			i_err = dynet::pickneglogsoftmax(i_r_t, next_words);

		v_errors.push_back(i_err);
	}

	dynet::Expression i_tloss = dynet::sum_batches(dynet::sum(v_errors));

	return i_tloss;
}

void TransformerLModel::get_avg_losses(dynet::ComputationGraph &cg
	, const WordIdSentences& tsents
	, std::vector<float>& v_losses
	, bool do_sum)
{
	// decode target
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, tsents);// ((num_units, Ly), batch_size)

	// get losses	
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859

	// compute the logit and linear projections
	dynet::Expression i_r = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_ctx});// ((|V_T|, (Ly-1)), batch_size)

	std::vector<dynet::Expression> v_errors;
	unsigned tlen = _decoder.get()->_batch_tlen;
	std::vector<unsigned> next_words(tsents.size());
	for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
		for(size_t bs = 0; bs < tsents.size(); bs++){
			next_words[bs] = (tsents[bs].size() > (t + 1)) ? (unsigned)tsents[bs][t + 1] : _tfc._sm._kTGT_EOS;
		}

		// get the prediction at timestep t
		//dynet::Expression i_r_t = dynet::select_cols(i_r, {t});// shifted right, ((|V_T|, 1), batch_size)
		dynet::Expression i_r_t = dynet::pick(i_r, t, 1);// shifted right, ((|V_T|, 1), batch_size)
	
		// log_softmax and loss
		dynet::Expression i_err = dynet::pickneglogsoftmax(i_r_t, next_words);

		v_errors.push_back(i_err);
	}

	dynet::Expression i_tloss;
       	if (!do_sum) i_tloss = dynet::sum(v_errors) / tlen;// loss normalised by max sequence length in batch
	else i_tloss = dynet::sum_batches(dynet::sum(v_errors)) / tlen;
	v_losses = dynet::as_vector(cg.incremental_forward(i_tloss));

	cg.clear();
}

dynet::Expression TransformerLModel::build_graph(dynet::ComputationGraph &cg
	, const std::vector<dynet::Expression>& v_soft_ssents
	, bool is_eval_on_dev)
{	
	// decode target
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, v_soft_ssents);// ((num_units, Ly), batch_size)

	// get losses	
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = _decoder.get()->get_wrd_embedding_matrix(cg);// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859

	// compute the logit and linear projections
	dynet::Expression i_r = dynet::affine_transform({i_Wo_bias, dynet::transpose(i_Wo_emb_tgt), i_tgt_ctx});// ((|V_T|, (Ly-1)), batch_size)

	std::vector<dynet::Expression> v_errors;
	unsigned tlen = _decoder.get()->_batch_tlen;
	for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
		// get the prediction at timestep t
		//dynet::Expression i_r_t = dynet::select_cols(i_r, {t});// shifted right, ((|V_T|, 1), batch_size)
		dynet::Expression i_r_t = dynet::pick(i_r, t, 1);// ((|V_T|, 1), batch_size)

		// log_softmax and loss
		dynet::Expression i_err;
		if (_tfc._use_label_smoothing && !is_eval_on_dev/*only applies in training*/)
		{// w/ label smoothing (according to section 7.5.1 of http://www.deeplearningbook.org/contents/regularization.html) and https://arxiv.org/pdf/1512.00567v1.pdf.
			// label smoothing regularizes a model based on a softmax with k output values by replacing the hard 0 and 1 classification targets with targets of \epsilon / (k−1) and 1 − \epsilon, respectively!
			dynet::Expression i_log_softmax = dynet::log_softmax(i_r_t);
			dynet::Expression i_pre_loss = (dynet::transpose(v_soft_ssents[t+1]) * i_Wo_emb_tgt) * (-i_log_softmax);
			dynet::Expression i_ls_loss = -dynet::sum_elems(i_log_softmax) / (_tfc._tgt_vocab_size - 1);// or -dynet::mean_elems(i_log_softmax)
			i_err = (1.f - _tfc._label_smoothing_weight) * i_pre_loss + _tfc._label_smoothing_weight * i_ls_loss;
		}
		else 
			i_err = (dynet::transpose(v_soft_ssents[t+1]) * i_Wo_emb_tgt) *  (-dynet::log_softmax(i_r_t));

		v_errors.push_back(i_err);
	}
	
	dynet::Expression i_tloss = dynet::sum_batches(dynet::sum(v_errors));

	return i_tloss;
}

void TransformerLModel::sample(dynet::ComputationGraph& cg, WordIdSentence &target, const std::string& prefix)
{
	_tfc._is_training = false;
	
	const int& sos_sym = _tfc._sm._kTGT_SOS;
	const int& eos_sym = _tfc._sm._kTGT_EOS;

	// start of sentence
	target.clear();
	target.push_back(sos_sym); 

	std::vector<std::string> pwords = split_words(prefix);
	for (auto& word : pwords) target.push_back(_dict.convert(word));

	std::vector<dynet::Expression> aligns;// FIXME: unused
	std::stringstream ss;
	ss << "<s>";
	unsigned t = 0;
	while (target.back() != eos_sym) 
	{		
		dynet::Expression ydist = this->step_forward(cg, WordIdSentences{1, target}/*non-batched*/, false, aligns);

		auto dist = dynet::as_vector(cg.incremental_forward(ydist));
		double p = rand01();
		WordId w = 0;
		for (; w < (WordId)dist.size(); ++w) {
			p -= dist[w];
			if (p < 0.f) break;
		}

		// this shouldn't happen
		if (w == (WordId)dist.size()) w = eos_sym;

		target.push_back(w);

		t += 1;
		if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

		cg.clear();
	}

	_tfc._is_training = true;
}

dynet::ParameterCollection& TransformerLModel::get_model_parameters(){
	return *_all_params.get();
}

void TransformerLModel::initialise_params_from_file(const std::string &params_file)
{
	dynet::load_dynet_model(params_file, _all_params.get());// FIXME: use binary streaming instead for saving disk spaces?
}

void TransformerLModel::save_params_to_file(const std::string &params_file)
{
	dynet::save_dynet_model(params_file, _all_params.get());// FIXME: use binary streaming instead for saving disk spaces?
}

void TransformerLModel::set_dropout(bool is_activated){
	_tfc._use_dropout = is_activated;
}

dynet::Dict& TransformerLModel::get_dict()
{
	return _dict;
}

TransformerConfig& TransformerLModel::get_config(){
	return _tfc;
}

//---

}; // namespace transformer



