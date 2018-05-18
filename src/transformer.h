/* This is an implementation of Transformer architecture from https://arxiv.org/abs/1706.03762 (Attention is All You need).
* Developed by Cong Duy Vu Hoang
* Updated: 1 Nov 2017
*/

#pragma once

// All utilities
#include "utils.h"

// Layers
#include "layers.h"

using namespace std;
using namespace dynet;

namespace transformer {

//--- Encoder Layer
struct EncoderLayer{
	explicit EncoderLayer(DyNetModel* mod, TransformerConfig& tfc)
		:_self_attention_sublayer(mod, tfc)
		, _feed_forward_sublayer(mod, tfc)
	{		
		// for layer normalisation
		_p_ln1_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln1_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
		_p_ln2_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
		_p_ln2_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));

		_p_tfc = &tfc;
	}

	~EncoderLayer(){}

	// multi-head attention sub-layer
	MultiHeadAttentionLayer _self_attention_sublayer;

	// position-wise feed forward sub-layer
	FeedForwardLayer _feed_forward_sublayer;

	// for layer normalisation
	dynet::Parameter _p_ln1_g, _p_ln1_b;// layer normalisation 1
	dynet::Parameter _p_ln2_g, _p_ln2_b;// layer normalisation 2

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;

	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const dynet::Expression& i_src
		, const MaskBase& self_mask)
	{	
		// get expressions for layer normalisation, e.g., i_ln1_g, i_ln1_b, i_ln2_g, i_ln2_b
		dynet::Expression i_ln1_g = dynet::parameter(cg, _p_ln1_g);
		dynet::Expression i_ln1_b = dynet::parameter(cg, _p_ln1_b);
		dynet::Expression i_ln2_g = dynet::parameter(cg, _p_ln2_g);
		dynet::Expression i_ln2_b = dynet::parameter(cg, _p_ln2_b);

		dynet::Expression i_encl = i_src;
		
		// multi-head attention sub-layer
		dynet::Expression i_mh_att = _self_attention_sublayer.build_graph(cg, i_encl, i_encl, self_mask);// ((num_units, Lx), batch_size)	

		// dropout to the above sub-layer
		if (_p_tfc->_use_dropout && _p_tfc->_encoder_sublayer_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_mh_att = dynet::dropout_dim(i_mh_att, 1/*col-major*/, _p_tfc->_encoder_sublayer_dropout_rate);// col-wise dropout
#else
			i_mh_att = dynet::dropout(i_mh_att, _p_tfc->_encoder_sublayer_dropout_rate);// full dropout
#endif

		// w/ residual connection
		i_encl = i_encl + i_mh_att;// ((num_units, Lx), batch_size)

		// position-wise layer normalisation 1
		i_encl = layer_norm_colwise_3(i_encl, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

		// position-wise feed-forward sub-layer
		dynet::Expression i_ff = _feed_forward_sublayer.build_graph(cg, i_encl);// ((num_units, Lx), batch_size)

		// w/ residual connection
		i_encl = i_encl + i_ff;// ((num_units, Lx), batch_size)

		// position-wise layer normalisation 2
		i_encl = layer_norm_colwise_3(i_encl, i_ln2_g, i_ln2_b);// ((num_units, Lx), batch_size)

		return i_encl;
	}
};

struct Encoder{
	explicit Encoder(DyNetModel* mod, TransformerConfig& tfc)
	{
		_p_embed_s = mod->add_lookup_parameters(tfc._src_vocab_size, {tfc._num_units});

		if (!tfc._use_hybrid_model && tfc._position_encoding == 1 && (tfc._position_encoding_flag == 0 || tfc._position_encoding_flag == 1)){
			_p_embed_pos = mod->add_lookup_parameters(tfc._max_length, {tfc._num_units});
		}

		if (tfc._use_hybrid_model){
			_v_p_src_rnns.resize(3);// first twos are forward and backward RNNs; third is forward RNN.
			for (unsigned l = 0; l < 3; l++){
				if (l == 2)
					_v_p_src_rnns[l].reset(new dynet::LSTMBuilder(1, tfc._num_units * 2, tfc._num_units, *mod, true/*w/ layer norm*/));
				else
					_v_p_src_rnns[l].reset(new dynet::LSTMBuilder(1, tfc._num_units, tfc._num_units, *mod, true/*w/ layer norm*/));
			}
		}

		for (unsigned l = 0; l < tfc._nlayers; l++){
			_v_enc_layers.push_back(EncoderLayer(mod, tfc));
		}

		_scale_emb = sqrt(tfc._num_units);

		_p_tfc = &tfc;
	}

	~Encoder(){}

	dynet::LookupParameter _p_embed_s;// source embeddings

	dynet::LookupParameter _p_embed_pos;// position embeddings
	unsigned _position_encoding = 1;

	// hybrid architecture: use LSTM-based RNNs over word embeddings instead of word embeddings + positional encodings
	std::vector<std::shared_ptr<dynet::LSTMBuilder>> _v_p_src_rnns;
	
	std::vector<EncoderLayer> _v_enc_layers;// stack of identical encoder layers

	// --- intermediate variables
	float _scale_emb = 0.f;
	unsigned _batch_slen = 0;
	MaskBase _self_mask;
	// ---

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;

	dynet::Expression get_wrd_embedding_matrix(dynet::ComputationGraph &cg){
		return dynet::parameter(cg, _p_embed_s);// target word embedding matrix (num_units x |V_S|)
	}
	
	dynet::Expression get_wrd_embeddings(dynet::ComputationGraph& cg, const std::vector<unsigned>& words){
		return dynet::lookup(cg, _p_embed_s, words);
	}

	dynet::Expression compute_embeddings_and_masks(dynet::ComputationGraph &cg
		, const WordIdSentences& sents/*batch of sentences*/
		, ModelStats* pstats=nullptr)
	{
		// compute embeddings
		// get max length within a batch
		unsigned bsize = sents.size();
		size_t max_len = sents[0].size();
		for(size_t i = 1; i < bsize; i++) max_len = std::max(max_len, sents[i].size());
		_batch_slen = max_len;

		std::vector<dynet::Expression> source_embeddings;   
		std::vector<unsigned> words(bsize);
		std::vector<std::vector<float>> v_seq_masks(bsize);
		dynet::Expression i_src;
		if (_p_tfc->_use_hybrid_model){
			// first 2 layers (forward and backward)
			// run a RNN backward and forward over the source sentence
			// and stack the top-level hidden states from each model 
			// and feed them into yet another forward RNN as 
			// the representation at each position
			// inspired from Google NMT system (https://arxiv.org/pdf/1609.08144.pdf)

			// first RNN (forward)
			std::vector<Expression> src_fwd(max_len);
			_v_p_src_rnns[0]->new_graph(cg);
			_v_p_src_rnns[0]->start_new_sequence();
			for (unsigned l = 0; l < max_len; l++){
				for (unsigned bs = 0; bs < bsize; ++bs){
					//words[bs] = (l < sents[bs].size()) ? (unsigned)sents[bs][l] : (unsigned)_p_tfc->_sm._kSRC_EOS;		
					if (l < sents[bs].size()){ 
						words[bs] = (unsigned)sents[bs][l];

						if (pstats){
							pstats->_words_src++; 
							if (sents[bs][l] == _p_tfc->_sm._kSRC_UNK) pstats->_words_src_unk++;
						}

						v_seq_masks[bs].push_back(0.f);// padding position
					}
					else{
						words[bs] = (unsigned)_p_tfc->_sm._kSRC_EOS;
						v_seq_masks[bs].push_back(1.f);// padding position
					}
				}

				src_fwd[l] = _v_p_src_rnns[0]->add_input(dynet::lookup(cg, _p_embed_s, words));
			}

			// second RNN (backward)
			std::vector<Expression> src_bwd(max_len);
			_v_p_src_rnns[1]->new_graph(cg);
			_v_p_src_rnns[1]->start_new_sequence();
			for (int l = max_len - 1; l >= 0; --l) { // int instead of unsigned for negative value of l
				// offset by one position to the right, to catch </s> and generally
				// not duplicate the w_t already captured in src_fwd[t]
				for (unsigned bs = 0; bs < bsize; ++bs) 
					words[bs] = ((unsigned)l < sents[bs].size()) ? (unsigned)sents[bs][l] : (unsigned)_p_tfc->_sm._kSRC_EOS;
				src_bwd[l] = _v_p_src_rnns[1]->add_input(dynet::lookup(cg, _p_embed_s, words));
			}

			// third RNN (yet another forward)
			_v_p_src_rnns[2]->new_graph(cg);
			_v_p_src_rnns[2]->start_new_sequence();
			for (unsigned l = 0; l < max_len; l++){
				source_embeddings.push_back(_v_p_src_rnns[2]->add_input(dynet::concatenate(std::vector<dynet::Expression>({src_fwd[l], src_bwd[l]}))));
			}
			
			i_src = dynet::concatenate_cols(source_embeddings);
		}
		else{
			// source embeddings
			for (unsigned l = 0; l < max_len; l++){
				for (unsigned bs = 0; bs < bsize; ++bs){
					//words[bs] = (l < sents[bs].size()) ? (unsigned)sents[bs][l] : (unsigned)_p_tfc->_sm._kSRC_EOS;		
					if (l < sents[bs].size()){ 
						words[bs] = (unsigned)sents[bs][l];

						if (pstats){
							pstats->_words_src++; 
							if (sents[bs][l] == _p_tfc->_sm._kSRC_UNK) pstats->_words_src_unk++;
						}

						v_seq_masks[bs].push_back(0.f);// padding position
					}
					else{
						words[bs] = (unsigned)_p_tfc->_sm._kSRC_EOS;
						v_seq_masks[bs].push_back(1.f);// padding position
					}
				}

				source_embeddings.push_back(dynet::lookup(cg, _p_embed_s, words));
			}
			i_src = dynet::concatenate_cols(source_embeddings);// ((num_units, Lx), batch_size)

			i_src = i_src * _scale_emb;// scaled embeddings

			// + postional encoding
			if (_p_tfc->_position_encoding_flag == 0 || _p_tfc->_position_encoding_flag == 1){
				if (_p_tfc->_position_encoding == 1){// learned positional embedding 
					std::vector<dynet::Expression> pos_embeddings;  
					std::vector<unsigned> positions(bsize);
					for (unsigned l = 0; l < max_len; l++){
						for (unsigned bs = 0; bs < bsize; ++bs){
							if (l >= _p_tfc->_max_length) positions[bs] = _p_tfc->_max_length - 1;// Trick: if using learned position encoding, during decoding/inference, sentence length may be longer than fixed max length. We overcome this by tying to (_p_tfc._max_length - 1).
							else
								positions[bs] = l;
					}

						pos_embeddings.push_back(dynet::lookup(cg, _p_embed_pos, positions));
					}
					dynet::Expression i_pos = dynet::concatenate_cols(pos_embeddings);// ((num_units, Lx), batch_size)

					i_src = i_src + i_pos;
				}
				else if (_p_tfc->_position_encoding == 2){// sinusoidal positional encoding
					dynet::Expression i_pos = make_sinusoidal_position_encoding(cg, i_src.dim());

					i_src = i_src + i_pos;
				}
				else if (_p_tfc->_position_encoding != 0) TRANSFORMER_RUNTIME_ASSERT("Unknown positional encoding type!");
			}
		}	

		// dropout to the sums of the embeddings and the positional encodings
		if (_p_tfc->_use_dropout && _p_tfc->_encoder_emb_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_src = dynet::dropout_dim(i_src, 1/*col-major*/, _p_tfc->_encoder_emb_dropout_rate);// col-wise dropout
#else
			i_src = dynet::dropout(i_src, _p_tfc->_encoder_emb_dropout_rate);// full dropout
#endif

		// create maskings
		_self_mask.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
		_self_mask.create_padding_positions_masks(_p_tfc->_nheads);
#else
		_self_mask.create_padding_positions_masks(1);
#endif

		return i_src;
	}

	dynet::Expression compute_embeddings_and_masks(dynet::ComputationGraph &cg
		, const std::vector<dynet::Expression>& v_soft_sources/*batched soft sources*/
		, ModelStats* pstats=nullptr)
	{
		// compute embeddings
		// get max length within a batch
		unsigned bsize = v_soft_sources[0].dim().batch_elems();		
		_batch_slen = v_soft_sources.size();

		dynet::Expression i_src;
		if (_p_tfc->_use_hybrid_model){
			// first 2 layers (forward and backward)
			// run a RNN backward and forward over the source sentence
			// and stack the top-level hidden states from each model 
			// and feed them into yet another forward RNN as 
			// the representation at each position
			// inspired from Google NMT system (https://arxiv.org/pdf/1609.08144.pdf)

			std::vector<dynet::Expression> source_embeddings;   

			// first RNN (forward)
			std::vector<Expression> src_fwd(_batch_slen);
			_v_p_src_rnns[0]->new_graph(cg);
			_v_p_src_rnns[0]->start_new_sequence();
			for (unsigned l = 0; l < _batch_slen; l++){
				src_fwd[l] = _v_p_src_rnns[0]->add_input(v_soft_sources[l]);
			}

			// second RNN (backward)
			std::vector<Expression> src_bwd(_batch_slen);
			_v_p_src_rnns[1]->new_graph(cg);
			_v_p_src_rnns[1]->start_new_sequence();
			for (int l = _batch_slen - 1; l >= 0; --l) { // int instead of unsigned for negative value of l
				// offset by one position to the right, to catch </s> and generally
				// not duplicate the w_t already captured in src_fwd[t]
				src_bwd[l] = _v_p_src_rnns[1]->add_input(v_soft_sources[l]);
			}

			// third RNN (yet another forward)
			_v_p_src_rnns[2]->new_graph(cg);
			_v_p_src_rnns[2]->start_new_sequence();
			for (unsigned l = 0; l < _batch_slen; l++){
				source_embeddings.push_back(_v_p_src_rnns[2]->add_input(dynet::concatenate(std::vector<dynet::Expression>({src_fwd[l], src_bwd[l]}))));
			}
			
			i_src = dynet::concatenate_cols(source_embeddings);
		}
		else{
			// source embeddings			
			//i_src = dynet::concatenate_cols(v_soft_sources);// ((num_units, Lx), batch_size)
			i_src = dynet::concatenate_cols(v_soft_sources);// ((|V_S|, Lx), batch_size)
			i_src = this->get_wrd_embedding_matrix(cg)/*num_units x |V_S|*/ * i_src;// ((num_units, Lx), batch_size)

			i_src = i_src * _scale_emb;// scaled embeddings

			// + postional encoding
			if (_p_tfc->_position_encoding_flag == 0 || _p_tfc->_position_encoding_flag == 1){
				if (_p_tfc->_position_encoding == 1){// learned positional embedding 
					std::vector<dynet::Expression> pos_embeddings;  
					std::vector<unsigned> positions(bsize);
					for (unsigned l = 0; l < _batch_slen; l++){
						for (unsigned bs = 0; bs < bsize; ++bs){
							if (l >= _p_tfc->_max_length) positions[bs] = _p_tfc->_max_length - 1;// Trick: if using learned position encoding, during decoding/inference, sentence length may be longer than fixed max length. We overcome this by tying to (_p_tfc._max_length - 1).
							else
								positions[bs] = l;
					}

						pos_embeddings.push_back(dynet::lookup(cg, _p_embed_pos, positions));
					}
					dynet::Expression i_pos = dynet::concatenate_cols(pos_embeddings);// ((num_units, Lx), batch_size)

					i_src = i_src + i_pos;
				}
				else if (_p_tfc->_position_encoding == 2){// sinusoidal positional encoding
					dynet::Expression i_pos = make_sinusoidal_position_encoding(cg, i_src.dim());

					i_src = i_src + i_pos;
				}
				else if (_p_tfc->_position_encoding != 0) TRANSFORMER_RUNTIME_ASSERT("Unknown positional encoding type!");
			}
		}	

		// dropout to the sums of the embeddings and the positional encodings
		if (_p_tfc->_use_dropout && _p_tfc->_encoder_emb_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_src = dynet::dropout_dim(i_src, 1/*col-major*/, _p_tfc->_encoder_emb_dropout_rate);// col-wise dropout
#else
			i_src = dynet::dropout(i_src, _p_tfc->_encoder_emb_dropout_rate);// full dropout
#endif

		// create maskings
		std::vector<std::vector<float>> v_seq_masks(bsize, std::vector<float>(_batch_slen, 0.f));
		_self_mask.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
		_self_mask.create_padding_positions_masks(_p_tfc->_nheads);
#else
		_self_mask.create_padding_positions_masks(1);
#endif

		return i_src;
	}

	dynet::Expression build_graph(dynet::ComputationGraph &cg, const WordIdSentences& ssents/*batch of sentences*/, ModelStats* pstats=nullptr){
		// compute source (+ postion) embeddings
		dynet::Expression i_src_rep = compute_embeddings_and_masks(cg, ssents, pstats);// ((num_units, Lx), batch_size)
		
		// compute stacked encoder layers
		dynet::Expression i_enc_l_out = i_src_rep;
		for (auto& enc : _v_enc_layers){
			// stacking approach
			i_enc_l_out = enc.build_graph(cg, i_enc_l_out, _self_mask);// each position in the encoder can attend to all positions in the previous layer of the encoder.
		}

		return i_enc_l_out;// ((num_units, Lx), batch_size)
	}

	dynet::Expression build_graph(dynet::ComputationGraph &cg, const std::vector<dynet::Expression>& v_soft_sources/*batched soft sources*/, ModelStats* pstats=nullptr){
		// compute source (+ postion) embeddings
		dynet::Expression i_src_rep = compute_embeddings_and_masks(cg, v_soft_sources, pstats);// ((num_units, Lx), batch_size)
		
		// compute stacked encoder layers
		dynet::Expression i_enc_l_out = i_src_rep;
		for (auto& enc : _v_enc_layers){
			// stacking approach
			i_enc_l_out = enc.build_graph(cg, i_enc_l_out, _self_mask);// each position in the encoder can attend to all positions in the previous layer of the encoder.
		}

		return i_enc_l_out;// ((num_units, Lx), batch_size)
	}
};
typedef std::shared_ptr<Encoder> EncoderPointer;
//---

//--- Decoder Layer
struct DecoderLayer{
	explicit DecoderLayer(DyNetModel* mod, TransformerConfig& tfc)
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

		_p_tfc = &tfc;
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

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;

	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const dynet::Expression& i_enc_inp
		, const dynet::Expression& i_dec_inp
		, const MaskBase& self_mask
		, const MaskBase& src_mask)
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

		// multi-head source attention sub-layer
		dynet::Expression i_mh_src_att = _src_attention_sublayer.build_graph(cg, i_decl, i_enc_inp, src_mask);// ((num_units, Ly), batch_size)

		// dropout to the output of sub-layer
		if (_p_tfc->_use_dropout && _p_tfc->_decoder_sublayer_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_mh_src_att = dynet::dropout_dim(i_mh_src_att, 1/*col-major*/, _p_tfc->_decoder_sublayer_dropout_rate);// col-wise dropout
#else
			i_mh_src_att = dynet::dropout(i_mh_src_att, _p_tfc->_decoder_sublayer_dropout_rate);// full dropout
#endif

		// w/ residual connection
		i_decl = i_decl + i_mh_src_att;

		// layer normalisation 2
		i_decl = layer_norm_colwise_3(i_decl, i_ln2_g, i_ln2_b);// ((num_units, Ly), batch_size)

		// position-wise feed-forward sub-layer
		dynet::Expression i_ff = _feed_forward_sublayer.build_graph(cg, i_decl);// ((num_units, Ly), batch_size)

		// w/ residual connection
		i_decl = i_decl + i_ff;

		// layer normalisation 3
		i_decl = layer_norm_colwise_3(i_decl, i_ln3_g, i_ln3_b);// ((num_units, Ly), batch_size)

		return i_decl;
	}
};

struct Decoder{
	explicit Decoder(DyNetModel* mod, TransformerConfig& tfc, Encoder* p_encoder)
	{
		if (tfc._shared_embeddings)
			_p_embed_t = p_encoder->_p_embed_s;// use shared embeddings with source
		else
			_p_embed_t = mod->add_lookup_parameters(tfc._tgt_vocab_size, {tfc._num_units});

		if (!tfc._use_hybrid_model && tfc._position_encoding == 1 && (tfc._position_encoding_flag == 0 || tfc._position_encoding_flag == 2)){
			_p_embed_pos = mod->add_lookup_parameters(tfc._max_length, {tfc._num_units});
		}

		for (unsigned l = 0; l < tfc._nlayers; l++){
			_v_dec_layers.push_back(DecoderLayer(mod, tfc));
		}

		if (tfc._use_hybrid_model){
			_p_tgt_rnn.reset(new dynet::LSTMBuilder(1/*shallow*/, tfc._num_units, tfc._num_units, *mod, true/*w/ layer norm*/));
		}

		_scale_emb = std::sqrt(tfc._num_units);

		_p_tfc = &tfc;

		_p_encoder = p_encoder;
	}

	~Decoder(){}

	dynet::LookupParameter _p_embed_t;// source embeddings
	dynet::LookupParameter _p_embed_pos;// position embeddings

	// hybrid architecture: use LSTM-based RNN over word embeddings instead of word embeddings + positional encodings
	std::shared_ptr<dynet::LSTMBuilder> _p_tgt_rnn;

	std::vector<DecoderLayer> _v_dec_layers;// stack of identical decoder layers

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;

	// encoder object pointer
	Encoder* _p_encoder = nullptr;

	// --- intermediate variables
	float _scale_emb = 0.f;
	unsigned _batch_tlen = 0;
	// decoder masks
	MaskBase _self_mask;
	MaskBase _src_mask;
	// ---

	dynet::Expression get_wrd_embedding_matrix(dynet::ComputationGraph &cg){
		return dynet::parameter(cg, _p_embed_t);// target word embedding matrix (num_units x |V_T|)
	}

	dynet::Expression get_wrd_embeddings(dynet::ComputationGraph& cg, const std::vector<unsigned>& words){
		return dynet::lookup(cg, _p_embed_t, words);
	}

	dynet::Expression compute_embeddings_and_masks(dynet::ComputationGraph &cg
		, const WordIdSentences& sents/*batch of target sentences*/)
	{
		// compute embeddings			
		// get max length in a batch
		unsigned bsize = sents.size();
		size_t max_len = sents[0].size();
		for(size_t i = 1; i < bsize; i++) max_len = std::max(max_len, sents[i].size());
		_batch_tlen = max_len;

		std::vector<dynet::Expression> target_embeddings;   
		std::vector<unsigned> words(bsize);
		std::vector<std::vector<float>> v_seq_masks(bsize);
		dynet::Expression i_tgt;
		if (_p_tfc->_use_hybrid_model){
			// target embeddings via RNN
			_p_tgt_rnn->new_graph(cg);
			_p_tgt_rnn->start_new_sequence();
			for (unsigned l = 0; l < max_len - (_p_tfc->_is_training)?1:0; l++){// offset by 1 during training
				for (unsigned bs = 0; bs < bsize; ++bs)
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
				for (unsigned bs = 0; bs < bsize; ++bs)
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
			if (_p_tfc->_position_encoding_flag == 0 || _p_tfc->_position_encoding_flag == 2){
				if (_p_tfc->_position_encoding == 1){// learned positional embedding 
					std::vector<dynet::Expression> pos_embeddings;  
					std::vector<unsigned> positions(bsize);
					for (unsigned l = 0; l < max_len - (_p_tfc->_is_training)?1:0; l++){// offset by 1 during training
						for (unsigned bs = 0; bs < bsize; ++bs){
							if (l >= _p_tfc->_max_length) positions[bs] = _p_tfc->_max_length - 1;// Trick: if using learned position encoding, during decoding/inference, sentence length may be longer than fixed max length. We overcome this by tying to _p_tfc._max_length.
							else
								positions[bs] = l;
					}

						pos_embeddings.push_back(dynet::lookup(cg, _p_embed_pos, positions));
					}
					dynet::Expression i_pos = dynet::concatenate_cols(pos_embeddings);// ((num_units, Ly), batch_size)

					i_tgt = i_tgt + i_pos;
				}
				else if (_p_tfc->_position_encoding == 2){// sinusoidal positional encoding
					dynet::Expression i_pos = make_sinusoidal_position_encoding(cg, i_tgt.dim());

					i_tgt = i_tgt + i_pos;
				}
				else if (_p_tfc->_position_encoding != 0) TRANSFORMER_RUNTIME_ASSERT("Unknown positional encoding type!");
			}
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
		// source-attention
		_src_mask.create_seq_mask_expr(cg, v_seq_masks, false);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
		_src_mask.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, _p_tfc->_nheads);
#else 
		_src_mask.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, 1);
#endif

		return i_tgt;
	}

	// this applies for stochastic decoding only!
	dynet::Expression compute_embeddings_and_masks(dynet::ComputationGraph &cg
		, const std::vector<dynet::Expression>& v_soft_targets/*batch of soft word target embeddings*/)
	{
		// compute embeddings			
		// get max length in a batch
		unsigned bsize = v_soft_targets[0].dim().batch_elems();
		_batch_tlen = v_soft_targets.size();
		
		dynet::Expression i_tgt;
		if (_p_tfc->_use_hybrid_model){
			// target embeddings via RNN
			std::vector<dynet::Expression> target_embeddings; 
			_p_tgt_rnn->new_graph(cg);
			_p_tgt_rnn->start_new_sequence();
			for (unsigned l = 0; l < _batch_tlen; l++){
				target_embeddings.push_back(_p_tgt_rnn->add_input(v_soft_targets[l]));
			}

			i_tgt = dynet::concatenate_cols(target_embeddings);// ((num_units, Ly), batch_size)
		}
		else{
			//i_tgt = dynet::concatenate_cols(v_soft_targets);// ((num_units, Ly), batch_size)
			i_tgt = dynet::concatenate_cols(v_soft_targets);// ((|V_T|, Ly), batch_size)
			i_tgt = this->get_wrd_embedding_matrix(cg)/*(num_units, |V_T|)*/ * i_tgt;// ((num_units, Ly), batch_size)

			// scale
			i_tgt = i_tgt * _scale_emb;// scaled embeddings

			// + postional encoding
			if (_p_tfc->_position_encoding_flag == 0 || _p_tfc->_position_encoding_flag == 2){
				if (_p_tfc->_position_encoding == 1){// learned positional embedding 
					std::vector<dynet::Expression> pos_embeddings;  
					std::vector<unsigned> positions(bsize);
					for (unsigned l = 0; l < _batch_tlen; l++){
						for (unsigned bs = 0; bs < bsize; ++bs){
							if (l >= _p_tfc->_max_length) positions[bs] = _p_tfc->_max_length - 1;// Trick: if using learned position encoding, during decoding/inference, sentence length may be longer than fixed max length. We overcome this by tying to _p_tfc._max_length.
							else
								positions[bs] = l;
					}

						pos_embeddings.push_back(dynet::lookup(cg, _p_embed_pos, positions));
					}
					dynet::Expression i_pos = dynet::concatenate_cols(pos_embeddings);// ((num_units, Ly), batch_size)

					i_tgt = i_tgt + i_pos;
				}
				else if (_p_tfc->_position_encoding == 2){// sinusoidal positional encoding
					dynet::Expression i_pos = make_sinusoidal_position_encoding(cg, i_tgt.dim());

					i_tgt = i_tgt + i_pos;
				}
				else if (_p_tfc->_position_encoding != 0) TRANSFORMER_RUNTIME_ASSERT("Unknown positional encoding type!");
			}
		}

		// dropout to the sums of the embeddings and the positional encodings
		if (_p_tfc->_use_dropout && _p_tfc->_decoder_emb_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_tgt = dynet::dropout_dim(i_tgt, 1/*col-major*/, _p_tfc->_decoder_emb_dropout_rate);// col-wise dropout
#else
			i_tgt = dynet::dropout(i_tgt, _p_tfc->_decoder_emb_dropout_rate);// full dropout		
#endif

		std::vector<std::vector<float>> v_seq_masks(bsize, std::vector<float>(_batch_tlen, 0.f));

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
		// source-attention
		_src_mask.create_seq_mask_expr(cg, v_seq_masks, false);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
		_src_mask.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, _p_tfc->_nheads);
#else 
		_src_mask.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, 1);
#endif

		return i_tgt;
	}
	
	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const WordIdSentences& tsents/*batch of sentences*/
		, const dynet::Expression& i_src_rep)
	{		
		// compute target (+ postion) embeddings
		dynet::Expression i_tgt_rep = compute_embeddings_and_masks(cg, tsents);// ((num_units, Ly), batch_size)
	
		// compute the decoder representation		
		dynet::Expression i_dec_l_out = i_tgt_rep;
		for (auto& dec : _v_dec_layers){
			// stacking approach
			i_dec_l_out = dec.build_graph(cg, i_src_rep, i_dec_l_out, _self_mask, _src_mask);// each position in the decoder can attend to all positions (up to and including the current position) in the previous layer of the decoder.
		}
	
		return i_dec_l_out;// ((num_units, Ly), batch_size)
	}

	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const std::vector<dynet::Expression>& v_soft_targets/*batch of expressions*/
		, const dynet::Expression& i_src_rep)
	{		
		// compute target (+ postion) embeddings
		dynet::Expression i_tgt_rep = compute_embeddings_and_masks(cg, v_soft_targets);// ((num_units, Ly), batch_size)
	
		// compute the decoder representation		
		dynet::Expression i_dec_l_out = i_tgt_rep;
		for (auto& dec : _v_dec_layers){
			// stacking approach
			i_dec_l_out = dec.build_graph(cg, i_src_rep, i_dec_l_out, _self_mask, _src_mask);// each position in the decoder can attend to all positions (up to and including the current position) in the previous layer of the decoder.
		}
	
		return i_dec_l_out;// ((num_units, Ly), batch_size)
	}
};
typedef std::shared_ptr<Decoder> DecoderPointer;
//---

//--- Transformer Model
struct TransformerModel {

public:
	explicit TransformerModel(const TransformerConfig& tfc, dynet::Dict& sd, dynet::Dict& td);

	explicit TransformerModel();

	~TransformerModel(){}

	// for training
	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const WordIdSentences& ssents/*batched*/
		, const WordIdSentences& tsents/*batched*/
		, ModelStats* pstats=nullptr
		, bool is_eval_on_dev=false);
	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const WordIdSentences& ssents
		, const WordIdSentences& tsents
		, const dynet::Expression& i_rl // additional reward for current loss, usually with shape=((1, 1), batch_size)
		, ModelStats* pstats=nullptr
		, bool is_eval_on_dev=false);
	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const std::vector<dynet::Expression>& v_soft_ssents/*batched*/
		, const WordIdSentences& tsents/*batched*/
		, ModelStats* pstats=nullptr
		, bool is_eval_on_dev=false);
	// for decoding
	dynet::Expression compute_source_rep(dynet::ComputationGraph &cg
		, const WordIdSentences& ssents);// source representation given real sources
	dynet::Expression step_forward(dynet::ComputationGraph & cg
		, const dynet::Expression& i_src_rep
		, const WordIdSentence &partial_sent
		, bool log_prob
		, std::vector<dynet::Expression> &aligns
		, float sm_temp=1.f);// forward step to get softmax scores
	dynet::Expression step_forward(dynet::ComputationGraph &cg
		, const dynet::Expression& i_src_rep
		, const WordIdSentences &partial_sents
		, bool log_prob
		, std::vector<dynet::Expression> &aligns
		, float sm_temp=1.f);
	dynet::Expression step_forward(dynet::ComputationGraph &cg
		, const dynet::Expression& i_src_rep
		, std::vector<Expression>& v_soft_targets
		, float sm_temp=1.f);
	void sample(dynet::ComputationGraph& cg, const WordIdSentence &source, WordIdSentence &target);// random sampling
	void sample(dynet::ComputationGraph& cg, const WordIdSentences &sources, WordIdSentences &targets); // batched version of random sampling
	void sample_sentences(dynet::ComputationGraph& cg
	        , const WordIdSentence& source
        	, unsigned num_samples
	        , std::vector<WordIdSentence>& samples
		, std::vector<float>& v_probs);
	void greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, WordIdSentence &target);// greedy decoding
	void greedy_decode(dynet::ComputationGraph& cg, const WordIdSentences &sources, WordIdSentences &targets);// batched version of greedy decoding
	//void beam_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, WordIdSentence &target, unsigned beam_width);// beam search decoding (return one best hypo)
	void beam_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, WordIdSentences &targets, unsigned beam_width, unsigned top_beams=1);// beam search decoding (return top_beams hypos)
	void stochastic_decode(dynet::ComputationGraph& cg, const WordIdSentences &sources, unsigned length_ratio, std::vector<Expression>& v_soft_targets); // batched stochastic decoding

	dynet::ParameterCollection& get_model_parameters();
	void initialise_params_from_file(const std::string &params_file);
	void save_params_to_file(const std::string &params_file);

	void set_dropout(bool is_activated = true);

	dynet::Dict& get_source_dict();
	dynet::Dict& get_target_dict();

	TransformerConfig& get_config();

protected:

	DyNetModelPointer _all_params;// all model parameters live in this object pointer. This object will be automatically released once unused!

	EncoderPointer _encoder;// encoder

	DecoderPointer _decoder;// decoder

	std::pair<dynet::Dict, dynet::Dict> _dicts;// pair of source and target vocabularies

	dynet::Parameter _p_Wo_bias;// bias of final linear projection layer

	TransformerConfig _tfc;// local configuration storage
};

TransformerModel::TransformerModel(){
	_all_params = nullptr;
	_encoder = nullptr;
	_decoder = nullptr;
}

TransformerModel::TransformerModel(const TransformerConfig& tfc, dynet::Dict& sd, dynet::Dict& td)
: _tfc(tfc)
{
	_all_params.reset(new DyNetModel());// create new model parameter object

	_encoder.reset(new Encoder(_all_params.get(), _tfc));// create new encoder object

	_decoder.reset(new Decoder(_all_params.get(), _tfc, _encoder.get()));// create new decoder object

	// final output projection layer
	_p_Wo_bias = _all_params.get()->add_parameters({_tfc._tgt_vocab_size});// optional

	// dictionaries
	_dicts.first = sd;
	_dicts.second = td;
}

dynet::Expression TransformerModel::compute_source_rep(dynet::ComputationGraph &cg
	, const WordIdSentences& ssents)// for decoding only
{
	// encode source
	return _encoder.get()->build_graph(cg, ssents);// ((num_units, Lx), batch_size)
}

dynet::Expression TransformerModel::step_forward(dynet::ComputationGraph &cg
	, const dynet::Expression& i_src_rep
	, const WordIdSentence &partial_sent
	, bool log_prob
	, std::vector<dynet::Expression> &aligns
	, float sm_temp)
{
	// decode target
	// IMPROVEMENT: during decoding, some parts in partial_sent will be recomputed. This is wasteful, especially for beam search decoding.
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, WordIdSentences(1, partial_sent), i_src_rep);// the whole matrix of context representation for every words in partial_sent - which is also wasteful because we only need the representation of last comlumn?

	// only consider the prediction of last column in the matrix
	dynet::Expression i_tgt_t;
	if (partial_sent.size() == 1) i_tgt_t = i_tgt_ctx;
	else 
		//i_tgt_t = dynet::select_cols(i_tgt_ctx, {(unsigned)(partial_sent.size() - 1)});
		i_tgt_t = dynet::pick(i_tgt_ctx, (unsigned)(partial_sent.size() - 1), 1);// shifted right, ((|V_T|, 1), batch_size)

	// output linear projections (w/ bias)
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859
	dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)

	// FIXME: get the alignments for visualisation
	// ToDo

	// compute softmax prediction
	if (log_prob)
		return dynet::log_softmax(i_r_t / sm_temp);// log_softmax w/ temperature
	else
		return dynet::softmax(i_r_t / sm_temp);// softmax w/ temperature
}

// batched version
dynet::Expression TransformerModel::step_forward(dynet::ComputationGraph &cg
	, const dynet::Expression& i_src_rep
	, const WordIdSentences &partial_sents
	, bool log_prob
	, std::vector<dynet::Expression> &aligns
	, float sm_temp)
{
	// decode target
	// IMPROVEMENT: during decoding, some parts in partial_sent will be recomputed. This is wasteful, especially for beam search decoding.
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, partial_sents, i_src_rep);// the whole matrix of context representation for every words in partial_sents is computed - which is also wasteful because we only need the representation of last comlumn?

	// only consider the prediction of last column in the matrix	
	dynet::Expression i_tgt_t;
	if (_decoder.get()->_batch_tlen == 1) i_tgt_t = i_tgt_ctx;
	else 
		//i_tgt_t = dynet::select_cols(i_tgt_ctx, {(unsigned)(partial_sent.size() - 1)});
		i_tgt_t = dynet::pick(i_tgt_ctx, (unsigned)(_decoder.get()->_batch_tlen - 1), 1);// shifted right, ((|V_T|, 1), batch_size)

	// output linear projections (w/ bias)
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859
	dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)

	// FIXME: get the alignments for visualisation
	// ToDo

	// compute softmax prediction (note: return a batch of softmaxes)
	if (log_prob)
		return dynet::log_softmax(i_r_t / sm_temp);// log_softmax w/ temperature
	else
		return dynet::softmax(i_r_t / sm_temp);// softmax w/ temperature
}

dynet::Expression TransformerModel::step_forward(dynet::ComputationGraph &cg
		, const dynet::Expression& i_src_rep
		, std::vector<Expression>& v_soft_targets
		, float sm_temp)
{
	// decode target
	// IMPROVEMENT: during decoding, some parts in partial_sent will be recomputed. This is wasteful, especially for beam search decoding.
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, v_soft_targets, i_src_rep);// the whole matrix of context representation for every words in partial_sents is computed - which is also wasteful because we only need the representation of last comlumn?

	// only consider the prediction of last column in the matrix	
	dynet::Expression i_tgt_t;
	if (_decoder.get()->_batch_tlen == 1) i_tgt_t = i_tgt_ctx;
	else 
		//i_tgt_t = dynet::select_cols(i_tgt_ctx, {(unsigned)(partial_sent.size() - 1)});
		i_tgt_t = dynet::pick(i_tgt_ctx, (unsigned)(_decoder.get()->_batch_tlen - 1), 1);// shifted right, ((|V_T|, 1), batch_size)

	// output linear projections (w/ bias)
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859
	dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)

	// compute softmax prediction (note: return a batch of softmaxes)
	return dynet::softmax(i_r_t / sm_temp);// softmax w/ temperature
}

dynet::Expression TransformerModel::build_graph(dynet::ComputationGraph &cg
	, const WordIdSentences& ssents
	, const WordIdSentences& tsents
	, ModelStats* pstats
	, bool is_eval_on_dev)
{
	// encode source
	dynet::Expression i_src_ctx = _encoder.get()->build_graph(cg, ssents, pstats);// ((num_units, Lx), batch_size)
	
	// decode target
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, tsents, i_src_ctx);// ((num_units, Ly), batch_size)

	// get losses	
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859

// both of the followings work well!
#ifndef USE_LINEAR_TRANSFORMATION_BROADCASTING 
	// Note: can be more efficient if using direct computing for i_tgt_ctx (e.g., use affine_transform)
	std::vector<dynet::Expression> v_errors;
	unsigned tlen = _decoder.get()->_batch_tlen;
	std::vector<unsigned> next_words(tsents.size());
	for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
		for(size_t bs = 0; bs < tsents.size(); bs++){
			next_words[bs] = (tsents[bs].size() > (t + 1)) ? (unsigned)tsents[bs][t + 1] : _tfc._sm._kTGT_EOS;
			if (tsents[bs].size() > t && pstats)
				pstats->_words_tgt++;
				if (tsents[bs][t] == _tfc._sm._kTGT_UNK) pstats->_words_tgt_unk++;
			}
		}

		// compute the logit
		//dynet::Expression i_tgt_t = dynet::select_cols(i_tgt_ctx, {t});// shifted right
		dynet::Expression i_tgt_t = dynet::pick(i_tgt_ctx, t, 1);// shifted right, ((|V_T|, 1), batch_size)

		// output linear projections
		dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)
	
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
#else // Note: this way is much faster!
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
			i_err = dynet::pickneglogsoftmax(i_r_t, next_words);// ((1, 1), batch_size)

		v_errors.push_back(i_err);
	}
#endif

	dynet::Expression i_tloss = dynet::sum_batches(dynet::sum(v_errors));

	return i_tloss;
}

dynet::Expression TransformerModel::build_graph(dynet::ComputationGraph &cg
	, const WordIdSentences& ssents
	, const WordIdSentences& tsents
	, const dynet::Expression& i_rl // additional reward for current loss, usually with shape=((1, 1), batch_size)
	, ModelStats* pstats
	, bool is_eval_on_dev)
{
	// encode source
	dynet::Expression i_src_ctx = _encoder.get()->build_graph(cg, ssents, pstats);// ((num_units, Lx), batch_size)
	
	// decode target
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, tsents, i_src_ctx);// ((num_units, Ly), batch_size)

	// get losses	
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859

// both of the followings work well!
#ifndef USE_LINEAR_TRANSFORMATION_BROADCASTING 
	// Note: can be more efficient if using direct computing for i_tgt_ctx (e.g., use affine_transform)
	std::vector<dynet::Expression> v_errors;
	unsigned tlen = _decoder.get()->_batch_tlen;
	std::vector<unsigned> next_words(tsents.size());
	for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
		for(size_t bs = 0; bs < tsents.size(); bs++){
			next_words[bs] = (tsents[bs].size() > (t + 1)) ? (unsigned)tsents[bs][t + 1] : _tfc._sm._kTGT_EOS;
			if (tsents[bs].size() > t && pstats)
				pstats->_words_tgt++;
				if (tsents[bs][t] == _tfc._sm._kTGT_UNK) pstats->_words_tgt_unk++;
			}
		}

		// compute the logit
		//dynet::Expression i_tgt_t = dynet::select_cols(i_tgt_ctx, {t});// shifted right
		dynet::Expression i_tgt_t = dynet::pick(i_tgt_ctx, t, 1);// shifted right, ((|V_T|, 1), batch_size)

		// output linear projections
		dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)
	
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
#else // Note: this way is much faster!
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
			i_err = (1.f - _tfc._label_smoothing_weight) * i_pre_loss + _tfc._label_smoothing_weight * i_ls_loss;// ((1, 1), batch_size)
		}
		else 
			i_err = dynet::pickneglogsoftmax(i_r_t, next_words);// ((1, 1), batch_size)

		v_errors.push_back(i_err);
	}
#endif

	dynet::Expression i_tloss = dynet::sum_batches(dynet::cmult(dynet::sum(v_errors), i_rl));// reinforced loss

	return i_tloss;
}

dynet::Expression TransformerModel::build_graph(dynet::ComputationGraph &cg
	, const std::vector<dynet::Expression>& v_soft_ssents/*batched soft sources*/
	, const WordIdSentences& tsents/*batched*/
	, ModelStats* pstats
	, bool is_eval_on_dev)
{
	// encode source
	dynet::Expression i_src_ctx = _encoder.get()->build_graph(cg, v_soft_ssents, pstats);// ((num_units, Lx), batch_size)
	
	// decode target
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, tsents, i_src_ctx);// ((num_units, Ly), batch_size)

	// get losses	
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859

// both of the followings work well!
#ifndef USE_LINEAR_TRANSFORMATION_BROADCASTING 
	// Note: can be more efficient if using direct computing for i_tgt_ctx (e.g., use affine_transform)
	std::vector<dynet::Expression> v_errors;
	unsigned tlen = _decoder.get()->_batch_tlen;
	std::vector<unsigned> next_words(tsents.size());
	for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
		for(size_t bs = 0; bs < tsents.size(); bs++){
			next_words[bs] = (tsents[bs].size() > (t + 1)) ? (unsigned)tsents[bs][t + 1] : _tfc._sm._kTGT_EOS;
			if (tsents[bs].size() > t && pstats)
				pstats->_words_tgt++;
				if (tsents[bs][t] == _tfc._sm._kTGT_UNK) pstats->_words_tgt_unk++;
			}
		}

		// compute the logit
		//dynet::Expression i_tgt_t = dynet::select_cols(i_tgt_ctx, {t});// shifted right
		dynet::Expression i_tgt_t = dynet::pick(i_tgt_ctx, t, 1);// shifted right, ((|V_T|, 1), batch_size)

		// output linear projections
		dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)
	
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
#else // Note: this way is much faster!
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
			i_err = dynet::pickneglogsoftmax(i_r_t, next_words);// ((1, 1), batch_size)

		v_errors.push_back(i_err);
	}
#endif

	dynet::Expression i_tloss = dynet::sum_batches(dynet::sum(v_errors));

	return i_tloss;
}

void TransformerModel::sample(dynet::ComputationGraph& cg, const WordIdSentence &source, WordIdSentence &target)
{
	_tfc._is_training = false;
	
	const int& sos_sym = _tfc._sm._kTGT_SOS;
	const int& eos_sym = _tfc._sm._kTGT_EOS;

	// start of sentence
	target.clear();
	target.push_back(sos_sym); 

	dynet::Expression i_src_rep = this->compute_source_rep(cg, WordIdSentences(1, source)/*pseudo batch (1)*/);// ToDo: batch decoding

	std::vector<dynet::Expression> aligns;// FIXME: unused
	std::stringstream ss;
	ss << "<s>";
	unsigned t = 0;
	while (target.back() != eos_sym) 
	{
		cg.checkpoint();
				
		dynet::Expression ydist = this->step_forward(cg, i_src_rep, target, false, aligns);
	
		auto dist = dynet::as_vector(cg.incremental_forward(ydist));
		double p = rand01();
		WordId w = 0;
		for (; w < (WordId)dist.size(); ++w) {
			p -= dist[w];
			if (p < 0.f) break;
		}

		// this shouldn't happen
		if (w == (WordId)dist.size()) w = eos_sym;

		if (t > TARGET_LENGTH_LIMIT_FACTOR * source.size())
			w = eos_sym;

		target.push_back(w);

		t += 1;
		if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

		cg.revert();
	}

	cg.clear();

	_tfc._is_training = true;
}

void TransformerModel::sample(dynet::ComputationGraph& cg, const WordIdSentences &sources, WordIdSentences &targets) // batched version of random sampling
{
	_tfc._is_training = false;

	unsigned bsize = sources.size();
	size_t max_src_len = sources[0].size();
	for (unsigned bs = 1; bs < bsize; bs++) max_src_len = std::max(max_src_len, sources[bs].size());
	
	const int& sos_sym = _tfc._sm._kTGT_SOS;
	const int& eos_sym = _tfc._sm._kTGT_EOS;

	// start of sentences
	targets.clear();
	targets.resize(bsize, WordIdSentence(1, sos_sym)); 

	dynet::Expression i_src_rep = this->compute_source_rep(cg, sources);// batched

	std::vector<dynet::Expression> aligns;// FIXME: unused
	unsigned t = 0;
	while (true) 
	{
		cg.checkpoint();
				
		dynet::Expression ydist = this->step_forward(cg, i_src_rep, targets, false, aligns);// batched
		auto dist = dynet::as_vector(cg.incremental_forward(ydist));// bsize * TARGET_VOCAB_SIZE

		for (unsigned bs = 0 ; bs < bsize; bs++){
			double p = rand01();
			WordId w = 0;
			for (; w < (WordId)_tfc._tgt_vocab_size; ++w) {
				p -= dist[w + _tfc._tgt_vocab_size * bs];
				if (p < 0.f) break;
			}

			// this shouldn't happen
			if (w == (WordId)_tfc._tgt_vocab_size) w = eos_sym;

			if (t > TARGET_LENGTH_LIMIT_FACTOR * max_src_len)
				w = eos_sym;

			targets[bs].push_back(w);
		}

		t += 1;
		if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

		cg.revert();

		// to check stopping condition, e.g., all sampled sentences ended with </s>.
		bool stopped = true;
		for (unsigned bs = 0; bs < bsize; bs++){
			if (targets[bs].back() != eos_sym){
				stopped = false;
				break;
			}
		}

		if (stopped) break;
	}

	cg.clear();

	_tfc._is_training = true;
}

void TransformerModel::sample_sentences(dynet::ComputationGraph& cg
        , const WordIdSentence& source
        , unsigned num_samples
        , std::vector<WordIdSentence>& samples
	, std::vector<float>& v_probs)
{
	_tfc._is_training = false;
	
	const int& sos_sym = _tfc._sm._kTGT_SOS;
	const int& eos_sym = _tfc._sm._kTGT_EOS;
	
	dynet::Expression i_src_rep = this->compute_source_rep(cg, WordIdSentences(num_samples, source));
	
	std::vector<dynet::Expression> aligns;// FIXME: unused

	samples.clear();
	samples.resize(num_samples, WordIdSentence(1, sos_sym)); 
	std::vector<unsigned> words(num_samples, sos_sym);

	v_probs.clear();
	v_probs.resize(num_samples, 0.f);

	unsigned t = 0;
	while (true) 
	{
		cg.checkpoint();
	
		dynet::Expression i_ydist = this->step_forward(cg, i_src_rep, samples, false, aligns);// batched, ((TARGET_VOCAB_SIZE, 1), num_samples)
	
		auto ydist = dynet::as_vector(cg.incremental_forward(i_ydist));// (num_samples * TARGET_VOCAB_SIZE)
		for (unsigned s = 0 ; s < num_samples; s++){
			double p = rand01();
			WordId w = 0;
			for (; w < (WordId)_tfc._tgt_vocab_size; ++w) {
				p -= ydist[w + _tfc._tgt_vocab_size * s];
				if (p < 0.f) break;
			}

			// this shouldn't happen
			if (w == (WordId)_tfc._tgt_vocab_size) w = eos_sym;

			if (t > TARGET_LENGTH_LIMIT_FACTOR * source.size())
				w = eos_sym;

			samples[s].push_back(w);
			words[s] = (unsigned)w;
		}

		dynet::Expression i_log_pick = -dynet::log(dynet::pick(i_ydist, words));// ((1, 1), num_samples)
		auto probs = dynet::as_vector(cg.incremental_forward(i_log_pick));// num_samples
		unsigned id = 0;
		for (auto& prob : probs) v_probs[id++] += prob;

		t += 1;
		if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

		cg.revert();

		// to check stopping condition, e.g., all sampled sentences ended with </s>.
		bool stopped = true;
		for (unsigned s = 0; s < num_samples; s++){
			if (samples[s].back() != eos_sym){
				stopped = false;
				break;
			}
		}
		
		if (stopped) break;
	}

	cg.clear();

	_tfc._is_training = true;
}

void TransformerModel::greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, WordIdSentence &target)
{
	_tfc._is_training = false;
	
	const int& sos_sym = _tfc._sm._kTGT_SOS;
	const int& eos_sym = _tfc._sm._kTGT_EOS;

	// start of sentence
	target.clear();
	target.push_back(sos_sym); 

	dynet::Expression i_src_rep = this->compute_source_rep(cg, WordIdSentences(1, source)/*pseudo batch (1)*/);// ToDo: batch decoding
	
	std::vector<dynet::Expression> aligns;// FIXME: unused
	unsigned t = 0;
	while (target.back() != eos_sym) 
	{
		cg.checkpoint();
			
		dynet::Expression i_ydist = this->step_forward(cg, i_src_rep, target, false, aligns);
		auto ydist = dynet::as_vector(cg.incremental_forward(i_ydist));
	
		// find the argmax next word (greedy)
		unsigned w = 0;
		auto pr_w = ydist[w];
		for (unsigned x = 1; x < ydist.size(); ++x) {
			if (ydist[x] > pr_w) {
				w = x;
				pr_w = ydist[w];
			}
		}

		// break potential infinite loop
		if (t > TARGET_LENGTH_LIMIT_FACTOR * source.size()) {
			w = eos_sym;
			pr_w = ydist[w];
		}

		// Note: use pr_w if getting the probability of the generated sequence!

		target.push_back(w);

		t += 1;
		if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

		cg.revert();
	}

	cg.clear();

	_tfc._is_training = true;
}

void TransformerModel::greedy_decode(dynet::ComputationGraph& cg, const WordIdSentences &sources, WordIdSentences &targets) // batched version of greedy decoding
{
	_tfc._is_training = false;

	unsigned bsize = sources.size();
	size_t max_src_len = sources[0].size();
	for (unsigned bs = 1; bs < bsize; bs++) max_src_len = std::max(max_src_len, sources[bs].size());
	
	const int& sos_sym = _tfc._sm._kTGT_SOS;
	const int& eos_sym = _tfc._sm._kTGT_EOS;

	// start of sentences
	targets.clear();
	targets.resize(bsize, WordIdSentence(1, sos_sym)); 

	dynet::Expression i_src_rep = this->compute_source_rep(cg, sources);// batched
	
	std::vector<dynet::Expression> aligns;// FIXME: unused
	unsigned t = 0;
	while (true) 
	{
		cg.checkpoint();
			
		dynet::Expression i_ydist = this->step_forward(cg, i_src_rep, targets, false, aligns);// batched
		auto ydist = dynet::as_vector(cg.incremental_forward(i_ydist));// bsize * TARGET_VOCAB_SIZE
	
		// find the argmax next word (greedy)
		for (unsigned bs = 0; bs < bsize; bs++){
			unsigned w = 0;
			auto pr_w = ydist[w + _tfc._tgt_vocab_size * bs];
			for (unsigned x = 1; x < _tfc._tgt_vocab_size; ++x) {
				if (ydist[x + _tfc._tgt_vocab_size * bs] > pr_w) {
					w = x;
					pr_w = ydist[x + _tfc._tgt_vocab_size * bs];
				}
			}

			// break potential infinite loop
			if (t > TARGET_LENGTH_LIMIT_FACTOR * max_src_len) {
				w = eos_sym;
				pr_w = ydist[w];
			}

			targets[bs].push_back(w);
		}

		t += 1;
		if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

		cg.revert();

		// to check stopping condition, e.g., all sampled sentences ended with </s>.
		bool stopped = true;
		for (unsigned bs = 0; bs < bsize; bs++){
			if (targets[bs].back() != eos_sym){
				stopped = false;
				break;
			}
		}

		if (stopped) break;
	}

	cg.clear();

	_tfc._is_training = true;
}

void TransformerModel::stochastic_decode(dynet::ComputationGraph& cg, const WordIdSentences &sources, unsigned length_ratio, std::vector<Expression>& v_soft_targets) // batched stochastic decoding
{
	_tfc._is_training = false;
	set_dropout(false);

	unsigned bsize = sources.size();
	size_t max_src_len = sources[0].size();
	for (unsigned bs = 1; bs < bsize; bs++) max_src_len = std::max(max_src_len, sources[bs].size());
	size_t max_tgt_len = max_src_len * length_ratio;
	
	const int& sos_sym = _tfc._sm._kTGT_SOS;

	// start of sentences
	v_soft_targets.clear();
	//std::vector<unsigned> sos_targets(bsize, sos_sym); 
	//v_soft_targets.push_back(this->_decoder->get_wrd_embeddings(cg, sos_targets));
	v_soft_targets.push_back(one_hot(cg, _tfc._tgt_vocab_size, std::vector<unsigned>(bsize, sos_sym)));// |V| x 1

	dynet::Expression i_src_rep = this->compute_source_rep(cg, sources);// batched
	//dynet::Expression i_tgt_emb = this->_decoder->get_wrd_embedding_matrix(cg);// hidden_dim x |VT|
	
	std::vector<dynet::Expression> aligns;// FIXME: unused
	unsigned t = 0;
	while (t < max_tgt_len) 
	{
		//cg.checkpoint(); // cannot do checkpointing here because the soft targets need to be memorized!
			
		dynet::Expression i_ydist = this->step_forward(cg, i_src_rep, v_soft_targets);// batched
		//v_soft_targets.push_back(i_tgt_emb/*H x |V|*/ * i_ydist/*|V|*/);// H x 1
		v_soft_targets.push_back(i_ydist);// |V| x 1

		t += 1;
		if (_tfc._position_encoding == 1 && (t >= _tfc._max_length || t >= max_tgt_len)) break;// to prevent over-length sample in learned positional encoding

		//cg.revert();
	}

	set_dropout(true);
	_tfc._is_training = true;
}

struct Hypothesis {
	Hypothesis() {};

	Hypothesis(int tgt, float cst, std::vector<dynet::Expression> &al)
		: target({tgt}), cost(cst), aligns(al) {}

	Hypothesis(int tgt, float cst, Hypothesis &last, std::vector<dynet::Expression> &al)
		: target(last.target), cost(cst), aligns(al) {
		target.push_back(tgt);
	}

	std::vector<int> target;
	float cost;
	std::vector<dynet::Expression> aligns;
};

// A simplified version of beam search decoding (transformer-decode will use integrated ensemble decoding instead!)
void TransformerModel::beam_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, WordIdSentences &targets, unsigned beam_width, unsigned top_beams)
{
	_tfc._is_training = false;
	
	const int& sos_sym = _tfc._sm._kTGT_SOS;
	const int& eos_sym = _tfc._sm._kTGT_EOS;
	unsigned int vocab_size = _dicts.second.size();

	// get the source representation
	dynet::Expression i_src_rep = this->compute_source_rep(cg, WordIdSentences(1, source)/*pseudo batch (1)*/);// ToDo: batch decoding
	
	std::vector<dynet::Expression> aligns;// FIXME: unused

	std::vector<Hypothesis> chart;
	chart.push_back(Hypothesis(sos_sym, 0.0f, aligns));

	std::vector<unsigned int> vocab(boost::copy_range<std::vector<unsigned int>>(boost::irange(0u, vocab_size)));
	std::vector<Hypothesis> completed;

	for (unsigned steps = 0; completed.size() < beam_width && steps < TARGET_LENGTH_LIMIT_FACTOR * source.size(); ++steps) {
		std::vector<Hypothesis> new_chart;

		for (auto& hprev: chart) {
			cg.checkpoint();
		
			dynet::Expression i_ydist = this->step_forward(cg, i_src_rep, hprev.target, false, aligns);

			// find the top k best next words
			auto ydist = dynet::as_vector(cg.incremental_forward(i_ydist));
			std::partial_sort(vocab.begin(), vocab.begin() + beam_width, vocab.end(), 
				[&ydist](unsigned int v1, unsigned int v2) { return ydist[v1] > ydist[v2]; });

			// add to chart
			for (auto vi = vocab.begin(); vi < vocab.begin() + beam_width; ++vi) {
				//if (new_chart.size() < beam_width) {
					Hypothesis hnew(*vi, hprev.cost - std::log(ydist[*vi]), hprev, aligns);
					if (*vi == (unsigned int)eos_sym){
						if (hnew.target.size() > 1) // to avoid bad sequences, e.g., <s> </s>
							completed.push_back(hnew);
					}
					else
						new_chart.push_back(hnew);
				//} 
			}
	
			cg.revert();
		}

		if (new_chart.size() > beam_width) {
			// sort new_chart by scores, to get kbest candidates
			std::partial_sort(new_chart.begin(), new_chart.begin() + beam_width, new_chart.end(),
				[](Hypothesis &h1, Hypothesis &h2) { return h1.cost < h2.cost; });
			new_chart.resize(beam_width);// only retain beam_width hypotheses
		}

		chart.swap(new_chart);
	}

	// If the model is too bad (at the beginning of training process), it will not be able to generate the well-formed sequences (e.g., with </s> at the end).
	// In this case, the generated sequences can be repeated and have the desired max length (e.g., 2 * source_len).
	// Here, completed.size() will be zero!
	if (completed.size() == 0)
		completed.swap(chart);// swap with current generated hypotheses
	
	// return 1-best hypo
	//auto best = std::min_element(completed.begin(), completed.end(),
	//		[](Hypothesis &h1, Hypothesis &h2) { return h1.cost/h1.target.size() < h2.cost/h2.target.size(); });// sort completed by their scores, adjusting for lengths -- not very effective, too short!
	//assert(best != completed.end());
	//target = best->target;

	// return top_beams hypos
	top_beams = (completed.size() > top_beams)? top_beams : completed.size();
	std::partial_sort(completed.begin(), completed.begin() + top_beams, completed.end(),
				[](Hypothesis &h1, Hypothesis &h2) { return h1.cost/h1.target.size() < h2.cost/h2.target.size(); });
	completed.resize(top_beams);

	targets.clear();
	for (auto& best : completed) targets.push_back(best.target);

	cg.clear();

	_tfc._is_training = true;
}

dynet::ParameterCollection& TransformerModel::get_model_parameters(){
	return *_all_params.get();
}

void TransformerModel::initialise_params_from_file(const std::string &params_file)
{
	dynet::load_dynet_model(params_file, _all_params.get());// FIXME: use binary streaming instead for saving disk spaces?
}

void TransformerModel::save_params_to_file(const std::string &params_file)
{
	dynet::save_dynet_model(params_file, _all_params.get());// FIXME: use binary streaming instead for saving disk spaces?
}

void TransformerModel::set_dropout(bool is_activated){
	_tfc._use_dropout = is_activated;
}

dynet::Dict& TransformerModel::get_source_dict()
{
	return _dicts.first;
}
dynet::Dict& TransformerModel::get_target_dict()
{
	return _dicts.second;
}

TransformerConfig& TransformerModel::get_config(){
	return _tfc;
}

//---

}; // namespace transformer



