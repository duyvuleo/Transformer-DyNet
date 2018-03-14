#pragma once

#include "def.h"

namespace transformer
{

// --- Sinusoidal Positional Encoding
// Note: think effective way to make this much faster!
dynet::Expression make_sinusoidal_position_encoding(dynet::ComputationGraph &cg, const dynet::Dim& dim);// global function
dynet::Expression make_sinusoidal_position_encoding(dynet::ComputationGraph &cg, const dynet::Dim& dim){
	unsigned nUnits = dim[0];
	unsigned nWords = dim[1];

	float num_timescales = nUnits / 2;
	float log_timescale_increment = std::log(10000.f) / (num_timescales - 1.f);

	std::vector<float> vSS(nUnits * nWords, 0.f);
	for(unsigned p = 0; p < nWords; ++p) {
		for(int i = 0; i < num_timescales; ++i) {
			float v = p * std::exp(i * -log_timescale_increment);
			vSS[p * nUnits + i] = std::sin(v);
			vSS[p * nUnits + num_timescales + i] = std::cos(v);
		}
	}

	return dynet::input(cg, {nUnits, nWords}, vSS);
}
// ---

//--- Simple Linear Layer (w/ or w/o bias)
struct LinearLayer{
	explicit LinearLayer(DyNetModel* mod, unsigned input_dim, unsigned output_dim, bool have_bias=true, bool initLC=false)
		: _have_bias(have_bias)
	{		
		_p_W = (initLC == false)?mod->add_parameters({output_dim, input_dim}):mod->add_parameters({output_dim, input_dim}, ParameterInitLeCunUniform(input_dim));
		if (_have_bias)
			_p_b = (initLC == false)?mod->add_parameters({output_dim}):mod->add_parameters({output_dim}, ParameterInitLeCunUniform(output_dim));
	}

	dynet::Expression apply(dynet::ComputationGraph& cg, const dynet::Expression& i_x, bool reconstruct_shape=true, bool time_distributed=false){
		dynet::Expression i_W = dynet::parameter(cg, _p_W);
		dynet::Expression i_b; 
		if (_have_bias)
			i_b = dynet::parameter(cg, _p_b);
	
		dynet::Expression i_x_in = (!time_distributed)?make_time_distributed(i_x)/*((input_dim, 1), batch_size * seq_len)*/:i_x/*((input_dim, seq_len), batch_size)*/;

		dynet::Expression i_x_out;
		if (_have_bias) i_x_out = dynet::affine_transform({i_b, i_W, i_x_in});// dim of i_x_out depends on i_x
		else i_x_out = i_W * i_x_in;

		if (!reconstruct_shape) return i_x_out;

		auto& d = i_x.dim();
		auto b = d.batch_elems();
		return make_reverse_time_distributed(i_x_out, d[1], b);// ((input_dim, seq_len), batch_size)
	}

	~LinearLayer(){}

	dynet::Parameter _p_W;
	dynet::Parameter _p_b;
	bool _have_bias = true;
};

//--- Highway Network Layer
/* Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wx + b)
    z = t * g(x) + (1 - t) * x
    where g is nonlinearity over x (e.g., a block of feedforward networks), t is transform gate, and (1 - t) is carry gate.
*/
struct HighwayNetworkLayer{
	explicit HighwayNetworkLayer(DyNetModel* mod, unsigned input_dim, bool have_bias=true)
#ifdef USE_LECUN_DIST_PARAM_INIT
		: _l_layer(mod, input_dim, input_dim, have_bias, true)
#else
		: _l_layer(mod, input_dim, input_dim, have_bias)
#endif
	{}

	dynet::Expression apply(dynet::ComputationGraph& cg
		, const dynet::Expression& i_x, const dynet::Expression& i_g_x
		, bool reconstruct_shape=true, bool time_distributed=false)
	{
		dynet::Expression i_l = _l_layer.apply(cg, i_x, reconstruct_shape, time_distributed);
		dynet::Expression i_t = dynet::logistic(i_l);
		dynet::Expression i_z = dynet::cmult(i_t, i_g_x) + dynet::cmult(1.f - i_t, i_x);
		return i_z;
	}

	LinearLayer _l_layer;

	~HighwayNetworkLayer(){}
};

struct FeedForwardLayer{
	explicit FeedForwardLayer(DyNetModel* mod, TransformerConfig& tfc)
		: _l_inner(mod, tfc._num_units, tfc._num_units * tfc._n_ff_units_factor/*4 by default according to the paper*/)
		, _l_outer(mod, tfc._num_units * tfc._n_ff_units_factor/*4 by default according to the paper*/, tfc._num_units)
	{		
		_p_tfc = &tfc;

		if (_p_tfc->_ffl_activation_type == FFL_ACTIVATION_TYPE::SWISH_LEARNABLE_BETA)
			_p_beta = mod->add_parameters({1});
	}	

	~FeedForwardLayer(){}	

	dynet::Parameter _p_beta;// learnable \beta for Swish activation function (work in progress!)

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;	

	dynet::Expression build_graph(dynet::ComputationGraph& cg, const dynet::Expression& i_inp/*num_units x L*/){
		// FFN(x) = relu(x * W1 + b1) * W2 + b2
		dynet::Expression i_inner = _l_inner.apply(cg, i_inp, false, true);// x * W1 + b1

		if (_p_tfc->_ffl_activation_type == FFL_ACTIVATION_TYPE::RELU)
			i_inner = dynet::rectify(i_inner);
		// use Swish from https://arxiv.org/pdf/1710.05941.pdf
		else if (_p_tfc->_ffl_activation_type == FFL_ACTIVATION_TYPE::SWISH) 
			i_inner = dynet::silu(i_inner);
		else if (_p_tfc->_ffl_activation_type == FFL_ACTIVATION_TYPE::SWISH_LEARNABLE_BETA){
			//dynet::Expression i_beta = dynet::parameter(cg, _p_beta);
			// FIXME: requires this: i_inner = dynet::silu(i_inner, i_beta); ? Not supported in DyNet yet!
			TRANSFORMER_RUNTIME_ASSERT("Feed-forward activation using Swish with learnable beta not implemented yet!");
		}
		else TRANSFORMER_RUNTIME_ASSERT("Unknown feed-forward activation type!");

		dynet::Expression i_outer = _l_outer.apply(cg, i_inner, false, true);// relu(x * W1 + b1) * W2 + b2

		// dropout for feed-forward layer
		// Note: this dropout can be moved to after RELU activation and before outer linear transformation (e.g., refers to Sockeye?).
		if (_p_tfc->_use_dropout && _p_tfc->_ff_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
			i_outer = dynet::dropout_dim(i_outer, 1/*col-major*/, _p_tfc->_ff_dropout_rate);
#else
			i_outer = dynet::dropout(i_outer, _p_tfc->_ff_dropout_rate);
#endif

		return i_outer;
	}

	LinearLayer _l_inner;
	LinearLayer _l_outer;
};

//---

// ---
// This MaskBase consists of all functions for maskings (both padding positions and future blinding)
struct MaskBase{
	explicit MaskBase(){}

	~MaskBase(){}

	void create_future_blinding_mask(dynet::ComputationGraph& cg, unsigned l){
		_i_mask_fb = create_triangle_mask(cg, l, false);	
	}

	void create_seq_mask_expr(dynet::ComputationGraph& cg
		, const std::vector<vector<float>>& v_seq_masks
		, bool self=true/*self-attention?*/)
	{
		unsigned l = v_seq_masks[0].size();

		std::vector<dynet::Expression> v_i_seq_masks;
		for (unsigned i = 0; i < v_seq_masks.size(); i++){
			dynet::Expression i_mask = (self) ? dynet::input(cg, {l, 1}, v_seq_masks[i]) : dynet::input(cg, {1, l}, v_seq_masks[i]);// ((l, 1), 1) or ((1, l), 1)
			v_i_seq_masks.push_back((self) ? i_mask * PSEUDO_MIN_VALUE : 1.f - i_mask);
		}
		_i_seq_mask = dynet::concatenate_to_batch(v_i_seq_masks);// ((l, 1), batch_size) or ((1, l), batch_size)
	}

	void create_padding_positions_masks(unsigned nheads) // for self-attention
	{
		unsigned l = _i_seq_mask.dim()[0];
		
		// key mask
		_i_mask_pp_k = dynet::concatenate_to_batch(std::vector<dynet::Expression>(nheads, dynet::concatenate_cols(std::vector<dynet::Expression>(l, _i_seq_mask))));// ((l, l), batch_size*nheads)
		
		// query mask
		_i_mask_pp_q = 1.f - _i_mask_pp_k / PSEUDO_MIN_VALUE;// ((l, l), batch_size*nheads)
	}

	void create_padding_positions_masks(const dynet::Expression& i_src_seq_mask, unsigned nheads) // for source-attention
	{
		unsigned ly = _i_seq_mask.dim()[1];
		unsigned lx = i_src_seq_mask.dim()[0];

		// key mask
		_i_mask_pp_k = dynet::concatenate_to_batch(std::vector<dynet::Expression>(nheads, dynet::concatenate_cols(std::vector<dynet::Expression>(ly, i_src_seq_mask))));// ((lx, ly), batch_size*nheads)

		// query mask
		_i_mask_pp_q = dynet::concatenate_to_batch(std::vector<dynet::Expression>(nheads, dynet::concatenate(std::vector<dynet::Expression>(lx, _i_seq_mask))));// ((lx, ly), batch_size*nheads)
	}

	// sequence mask
	dynet::Expression _i_seq_mask;

	// 2 masks for padding positions
	dynet::Expression _i_mask_pp_k;// for keys
	dynet::Expression _i_mask_pp_q;// for queries

	// 1 mask for future blinding
	dynet::Expression _i_mask_fb;
};
// ---

//--- Multi-Head Attention Layer
struct MultiHeadAttentionLayer{
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
	explicit MultiHeadAttentionLayer(DyNetModel* mod, TransformerConfig& tfc, bool is_future_blinding=false)
#ifdef USE_LECUN_DIST_PARAM_INIT
		: _l_W_Q(mod, tfc._num_units, tfc._num_units, false/*linear layer w/o bias*/, true)
		, _l_W_K(mod, tfc._num_units, tfc._num_units, false, true)
		, _l_W_V(mod, tfc._num_units, tfc._num_units, false, true)
		, _l_W_O(mod, tfc._num_units, tfc._num_units, false, true)
#else
		: _l_W_Q(mod, tfc._num_units, tfc._num_units, false/*linear layer w/o bias*/)
		, _l_W_K(mod, tfc._num_units, tfc._num_units, false)
		, _l_W_V(mod, tfc._num_units, tfc._num_units, false)
		, _l_W_O(mod, tfc._num_units, tfc._num_units, false)
#endif
	{
		_att_scale = 1.f / sqrt(tfc._num_units / tfc._nheads);

		_is_future_blinding = is_future_blinding;

		_p_tfc = &tfc;
	}

	~MultiHeadAttentionLayer(){}

	// linear projection matrices
	LinearLayer _l_W_Q;
	LinearLayer _l_W_K;
	LinearLayer _l_W_V;
	LinearLayer _l_W_O;// finishing linear layer

	// multi-head soft alignments 
	bool _use_soft_alignments = false;// will be necessary if soft alignments are used for visualisation or incorporation of additional regularisation techniques
	std::vector<dynet::Expression> _v_aligns;

	// attention scale factor
	float _att_scale = 0.f;

	bool _is_future_blinding = false;

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;

	void get_alignments(const dynet::Expression& i_aligns){
		_v_aligns.clear();
		_v_aligns = split_batch(i_aligns, _p_tfc->_nheads);// _v_aligns will have nheads of ((Lx, Ly),  batch_size)) expressions
	}

	dynet::Expression build_graph(dynet::ComputationGraph& cg
		, const dynet::Expression& i_y/*queries*/
		, const dynet::Expression& i_x/*keys and values. i_x is equal to i_y if using self_attention*/
		, const MaskBase& i_mask)
	{
		dynet::Expression i_Q = _l_W_Q.apply(cg, i_y, false, true);// ((num_units, Ly), batch_size)
		dynet::Expression i_K = _l_W_K.apply(cg, i_x, false, true);// ((num_units, Lx), batch_size)
		dynet::Expression i_V = _l_W_V.apply(cg, i_x, false, true);// ((num_units, Lx), batch_size)

		// Note: this will be done in parallel for efficiency!
		// e.g., utilising pseudo-batching
		dynet::Expression i_batch_Q = dynet::concatenate_to_batch(split_rows(i_Q, _p_tfc->_nheads));// ((num_units/nheads, Ly), batch_size*nheads)
		dynet::Expression i_batch_K = dynet::concatenate_to_batch(split_rows(i_K, _p_tfc->_nheads));// ((num_units/nheads, Lx), batch_size*nheads)
		dynet::Expression i_batch_V = dynet::concatenate_to_batch(split_rows(i_V, _p_tfc->_nheads));// ((num_units/nheads, Lx), batch_size*nheads)

		dynet::Expression i_atts;
		if (_p_tfc->_attention_type == ATTENTION_TYPE::DOT_PRODUCT){// Luong attention type
			dynet::Expression i_batch_alphas = (dynet::transpose(i_batch_K) * i_batch_Q) * _att_scale;// ((Lx, Ly),  batch_size*nheads)) (unnormalised) 

#ifdef USE_KEY_QUERY_MASKINGS
			// key masking
			i_batch_alphas = i_batch_alphas + i_mask._i_mask_pp_k;
#endif

			// future blinding masking
			if (_is_future_blinding)
				i_batch_alphas = dynet::softmax(i_batch_alphas + i_mask._i_mask_fb);// ((Lx, Ly),  batch_size*nheads)) (normalised, col-major)
			else
				i_batch_alphas = dynet::softmax(i_batch_alphas);// ((Lx, Ly),  batch_size*nheads)) (normalised, col-major)
			
#ifdef USE_KEY_QUERY_MASKINGS
			// query masking
			i_batch_alphas = dynet::cmult(i_batch_alphas, i_mask._i_mask_pp_q);// masked soft alignments
#endif

			// save the soft alignment in i_batch_alphas if necessary!
			if (_use_soft_alignments) get_alignments(i_batch_alphas);
					
			// attention dropout (col-major or full?)
			if (_p_tfc->_use_dropout && _p_tfc->_attention_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
				i_batch_alphas = dynet::dropout_dim(i_batch_alphas, 1/*col-major*/, _p_tfc->_attention_dropout_rate);// col-wise dropout
#else
				i_batch_alphas = dynet::dropout(i_batch_alphas, _p_tfc->_attention_dropout_rate);// full matrix
#endif

			i_batch_alphas = i_batch_V/*((num_units/nheads, Lx), batch_size*nheads)*/ * i_batch_alphas/*((Lx, Ly), batch_size*nheads))*/;// ((num_units/nheads, Ly), batch_size*nheads)

			i_atts = dynet::concatenate(split_batch(i_batch_alphas, _p_tfc->_nheads));// ((num_units, Ly), batch_size)			
		}
		else if (_p_tfc->_attention_type == ATTENTION_TYPE::ADDITIVE_MLP){// Bahdanau attention type
			TRANSFORMER_RUNTIME_ASSERT("MultiHeadAttentionLayer: Bahdanau attention type not yet implemented!");
		}
		else TRANSFORMER_RUNTIME_ASSERT("MultiHeadAttentionLayer: Unknown attention type!");
		
		// linear projection
		dynet::Expression i_proj_atts = _l_W_O.apply(cg, i_atts, false, true);// ((num_units, Ly), batch_size)

		return i_proj_atts;
	}
#else // without using pseudo-batching
	explicit MultiHeadAttentionLayer(DyNetModel* mod, TransformerConfig& tfc, bool is_future_blinding=false)
	{
		_p_WQ.resize(tfc._nheads);
		_p_WK.resize(tfc._nheads);
		_p_WV.resize(tfc._nheads);
		for (unsigned h = 0; h < tfc._nheads; h++){
			_p_WQ[h] = mod->add_parameters({tfc._num_units / tfc._nheads, tfc._num_units});// dk = num_units/nheads
			_p_WK[h] = mod->add_parameters({tfc._num_units / tfc._nheads, tfc._num_units});// dk
			_p_WV[h] = mod->add_parameters({tfc._num_units / tfc._nheads, tfc._num_units});// dv = num_units/nheads
		}

		_p_WO = mod->add_parameters({tfc._num_units, tfc._num_units});
		
		_att_scale = 1.f / sqrt(tfc._num_units / tfc._nheads);

		_is_future_blinding = is_future_blinding;

		_p_tfc = &tfc;
	}

	~MultiHeadAttentionLayer(){}

	// linear projection matrices
	std::vector<dynet::Parameter> _p_WQ;
	std::vector<dynet::Parameter> _p_WK;
	std::vector<dynet::Parameter> _p_WV;
	dynet::Parameter _p_WO;

	// multi-head soft alignments 
	bool _use_soft_alignments = false;// will be necessary if soft alignments are used for visualisation or incorporation of additional regularisation techniques
	std::vector<dynet::Expression> _v_aligns;

	// attention scale factor
	float _att_scale = 0.f;

	bool _is_future_blinding = false;

	// transformer config pointer
	TransformerConfig* _p_tfc = nullptr;

	dynet::Expression build_graph(dynet::ComputationGraph& cg
		, const dynet::Expression& i_y/*queries*/
		, const dynet::Expression& i_x/*keys and values. i_x is equal to i_y if using self_attention*/
		, const MaskBase& i_mask)
	{
		_v_aligns.clear();
		
		// Note: this should be done in parallel for efficiency!
		// e.g., utilising pseudo-batching?	
		std::vector<dynet::Expression> v_atts(_p_tfc->_nheads);
		for (unsigned h = 0; h < _p_tfc->_nheads; h++){
			dynet::Expression i_Q/*queries*/ = dynet::parameter(cg, _p_WQ[h])/*dk x num_units*/ * i_x/*num_units x Ly*/;// ((dk, Ly), batch_size)
			dynet::Expression i_K/*keys*/ = dynet::parameter(cg, _p_WK[h])/*dk x num_units*/ * i_y/*num_units x Lx*/;// ((dk, Lx), batch_size)
			dynet::Expression i_V/*values*/ = dynet::parameter(cg, _p_WV[h])/*dv x num_units*/ * i_y/*num_units x Lx*/;// ((dk, Lx), batch_size)

			dynet::Expression i_att_h;
			if (_p_tfc->_attention_type == ATTENTION_TYPE::DOT_PRODUCT){// Luong attention type
				dynet::Expression i_alpha_pre = (dynet::transpose(i_K) * i_Q) * _att_scale;// ((Lx, Ly), batch_size) (unnormalised) 

#ifdef USE_KEY_QUERY_MASKINGS
				// key masking
				i_alpha_pre = i_alpha_pre + i_mask._i_mask_pp_k);
#endif

				dynet::Expression i_alpha;
				if (_is_future_blinding)
					i_alpha = dynet::softmax(i_alpha_pre + i_mask._i_mask_fb);// ((Lx, Ly), batch_size) (normalised, col-major)
				else
					i_alpha = dynet::softmax(i_alpha_pre);// ((Lx, Ly), batch_size) (normalised, col-major)

#ifdef USE_KEY_QUERY_MASKINGS
				// query masking
				i_alpha = dynet::cmult(i_alpha, i_mask._i_mask_pp_q));// masked soft alignments
#endif

				// save the soft alignment in i_alpha if necessary!
				if (_use_soft_alignments) _v_aligns.push_back(i_alpha);
						
				// attention dropout (col-major or full?)
				if (_p_tfc->_use_dropout && _p_tfc->_attention_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
					i_alpha = dynet::dropout_dim(i_alpha, 1/*col-major*/, _p_tfc->_attention_dropout_rate);// col-wise dropout
#else
					i_alpha = dynet::dropout(i_alpha, _p_tfc->_attention_dropout_rate);// full dropout
#endif

				i_att_h = i_V * i_alpha;// ((dk, Ly), batch_size)
			}
			else if (_p_tfc->_attention_type == ATTENTION_TYPE::ADDITIVE_MLP){// Bahdanau attention type
				TRANSFORMER_RUNTIME_ASSERT("MultiHeadAttentionLayer: Bahdanau attention type not yet implemented!");
			}
			else TRANSFORMER_RUNTIME_ASSERT("MultiHeadAttentionLayer: Unknown attention type!");

			v_atts[h] = i_att_h;
		}

		// joint all head attentions
		dynet::Expression i_atts = dynet::concatenate(v_atts);// ((dk*nheads=num_units, Ly), batch_size)

		// linear projection
		dynet::Expression i_proj_atts = dynet::parameter(cg, _p_WO) * i_atts;// ((num_units, Ly), batch_size)

		return i_proj_atts;
	}
#endif
};
//---

};
