#include "dynet/dglstm.h"

#include "dynet/param-init.h"

#include <fstream>
#include <string>
#include <cassert>
#include <vector>
#include <iostream>

using namespace std;

//#define USE_STANDARD_LSTM_DEFINE 
/**
the standard definition has a forget gate computed using its own parameter. 
however, in dynet, the forget gate is computed as f_t = 1 - i_t, where i_t is the input gate. 
this is also used in LSTM.cc.

*/

namespace dynet {

enum { X2I, H2I, C2I, BI, 
#ifdef USE_STANDARD_LSTM_DEFINE
    X2F, H2F, C2F, BF, 
#endif
    X2O, H2O, C2O, BO, X2C, H2C, BC, X2K, C2K, Q2K, BK, STAB, X2K0 };

DGLSTMBuilder::DGLSTMBuilder(unsigned layers,
             unsigned input_dim,
             unsigned hidden_dim,
             ParameterCollection& model) : layers(layers), hid(hidden_dim) {
  Parameter p_x2k, p_c2k, p_q2k, p_bk, p_x2k0;
  unsigned layer_input_dim = input_dim;
  local_model = model.add_subcollection("dglstm-builder");

  input_dims = vector<unsigned>(layers, layer_input_dim);
  p_x2k0 = local_model.add_parameters({ hidden_dim, layer_input_dim });
  for (unsigned i = 0; i < layers; ++i) {
    input_dims[i] = layer_input_dim;
    // i
    Parameter p_x2i = local_model.add_parameters({ hidden_dim, layer_input_dim });
    Parameter p_h2i = local_model.add_parameters({ hidden_dim, hidden_dim });
    Parameter p_c2i = local_model.add_parameters({ hidden_dim, hidden_dim });
    Parameter p_bi = local_model.add_parameters({ hidden_dim });
#ifdef USE_STANDARD_LSTM_DEFINE
    // f
    Parameter p_x2f = local_model.add_parameters({ hidden_dim, layer_input_dim });
    Parameter p_h2f = local_model.add_parameters({ hidden_dim, hidden_dim });
    Parameter p_c2f = local_model.add_parameters({ hidden_dim, hidden_dim });
    Parameter p_bf = local_model.add_parameters({ hidden_dim });
#endif
    // o
    Parameter p_x2o = local_model.add_parameters({ hidden_dim, layer_input_dim });
    Parameter p_h2o = local_model.add_parameters({ hidden_dim, hidden_dim });
    Parameter p_c2o = local_model.add_parameters({ hidden_dim, hidden_dim });
    Parameter p_bo = local_model.add_parameters({ hidden_dim });

    // c
    Parameter p_x2c = local_model.add_parameters({ hidden_dim, layer_input_dim });
    Parameter p_h2c = local_model.add_parameters({ hidden_dim, hidden_dim });
    Parameter p_bc = local_model.add_parameters({ hidden_dim });

    p_x2k = local_model.add_parameters({ hidden_dim, layer_input_dim });
    p_c2k = local_model.add_parameters({ hidden_dim });
    p_bk = local_model.add_parameters({ hidden_dim });
    p_q2k = local_model.add_parameters({ hidden_dim });

    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    Parameter p_stab = local_model.add_parameters({ 1 });
    p_stab.zero();

    vector<Parameter> ps;
    if (i == 0)
        ps = { p_x2i, p_h2i, p_c2i, p_bi, 
#ifdef USE_STANDARD_LSTM_DEFINE
        p_x2f, p_h2f, p_c2f, p_bf, 
#endif
        p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc, p_x2k, p_c2k, p_q2k, p_bk, p_stab, p_x2k0 };
    else
        ps = { p_x2i, p_h2i, p_c2i, p_bi, 
#ifdef USE_STANDARD_LSTM_DEFINE
        p_x2f, p_h2f, p_c2f, p_bf, 
#endif
        p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc, p_x2k, p_c2k, p_q2k, p_bk, p_stab };

    params.push_back(ps);
  }  // layers

  dropout_rate = 0.f;  
}

void DGLSTMBuilder::initialize_biases(){
  biases.clear();
  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    Expression bimb = concatenate_cols(vector<Expression>(1, vars[BI]));
    Expression bcmb = concatenate_cols(vector<Expression>(1, vars[BC]));
    Expression bomb = concatenate_cols(vector<Expression>(1, vars[BO]));
    Expression bkmb = concatenate_cols(vector<Expression>(1, vars[BK]));
    Expression mc2kmb = concatenate_cols(vector<Expression>(1, vars[C2K]));
    Expression mq2kmb = concatenate_cols(vector<Expression>(1, vars[Q2K]));
#ifdef USE_STANDARD_LSTM_DEFINE
    Expression bfmb = concatenate_cols(vector<Expression>(1, vars[BF]));
#endif
    Expression i_stabilizer = exp(vars[STAB]);
    Expression i_stab = concatenate(vector<Expression>(input_dims[i], i_stabilizer));  /// self stabilizer

    vector<Expression> b = { bimb, bcmb, bomb, bkmb, mc2kmb, mq2kmb, i_stab 
#ifdef USE_STANDARD_LSTM_DEFINE
    , bfmb
#endif
    };

    biases.push_back(b);
  }    
}

void DGLSTMBuilder::new_graph_impl(ComputationGraph& cg, bool update){
  param_vars.clear();

  for (unsigned i = 0; i < layers; ++i){
    auto& p = params[i];

    //i
    Expression i_x2i = update ? parameter(cg,p[X2I]) : const_parameter(cg,p[X2I]);
    Expression i_h2i = update ? parameter(cg,p[H2I]) : const_parameter(cg,p[H2I]);
    Expression i_c2i = update ? parameter(cg,p[C2I]) : const_parameter(cg,p[C2I]);
    Expression i_bi = update ? parameter(cg,p[BI]) : const_parameter(cg,p[BI]);
#ifdef USE_STANDARD_LSTM_DEFINE
    //f
    Expression i_x2f = update ? parameter(cg, p[X2F]) : const_parameter(cg, p[X2F]);
    Expression i_h2f = update ? parameter(cg, p[H2F]) : const_parameter(cg, p[H2F]);
    Expression i_c2f = update ? parameter(cg, p[C2F]) : const_parameter(cg, p[C2F]);
    Expression i_bf = update ? parameter(cg, p[BF]) : const_parameter(cg, p[BF]);
#endif
    //o
    Expression i_x2o = update ? parameter(cg,p[X2O]) : const_parameter(cg,p[X2O]);
    Expression i_h2o = update ? parameter(cg,p[H2O]) : const_parameter(cg,p[H2O]);
    Expression i_c2o = update ? parameter(cg,p[C2O]) : const_parameter(cg,p[C2O]);
    Expression i_bo = update ? parameter(cg,p[BO]) : const_parameter(cg,p[BO]);
    //c
    Expression i_x2c = update ? parameter(cg,p[X2C]) : const_parameter(cg,p[X2C]);
    Expression i_h2c = update ? parameter(cg,p[H2C]) : const_parameter(cg,p[H2C]);
    Expression i_bc = update ? parameter(cg,p[BC]) : const_parameter(cg,p[BC]);

    vector<Expression> vars;

    //k
    Expression i_x2k = parameter(cg, p[X2K]);
    Expression i_q2k = parameter(cg, p[Q2K]);
    Expression i_c2k = parameter(cg, p[C2K]);
    Expression i_bk = parameter(cg, p[BK]);

    Expression i_stab = parameter(cg, p[STAB]);

    if (i == 0)
    {
        Expression i_x2k0 = parameter(cg, p[X2K0]);
        vars = { i_x2i, i_h2i, i_c2i, i_bi, 
#ifdef USE_STANDARD_LSTM_DEFINE
            i_x2f, i_h2f, i_c2f, i_bf, 
#endif
            i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc, i_x2k, i_c2k, i_q2k, i_bk, i_stab, i_x2k0 };
    }
    else
        vars = { i_x2i, i_h2i, i_c2i, i_bi, 
#ifdef USE_STANDARD_LSTM_DEFINE
        i_x2f, i_h2f, i_c2f, i_bf, 
#endif
        i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc, i_x2k, i_c2k, i_q2k, i_bk, i_stab };

    param_vars.push_back(vars);
  }

  _cg = &cg;

  initialize_biases();
}

// layout: 0..layers = c
//     layers+1..2*layers = h
void DGLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();

  if (hinit.size() > 0) {
    assert(layers*2 == hinit.size());
    h0.resize(layers);
    c0.resize(layers);
    for (unsigned i = 0; i < layers; ++i) {
      c0[i] = hinit[i];
      h0[i] = hinit[i + layers];
    }
    has_initial_state = true;
  } else {
    has_initial_state = false;
  }

  initialize_biases();
}

Expression DGLSTMBuilder::add_input_impl(int prev, const Expression& x) {
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();

  Expression lower_layer_c = x;  /// at layer 0, no lower memory but observation
  Expression in = x;
  Expression in_stb;

  int nutt = 1;

  for (unsigned i = 0; i < layers; ++i) {
    // apply dropout according to http://arxiv.org/pdf/1409.2329v5.pdf
    if (dropout_rate) in = dropout(in, dropout_rate);

    const vector<Expression>& vars = param_vars[i];
    Expression i_stabilizer = biases[i][6];
    Expression i_v_stab = concatenate_cols(vector<Expression>(nutt, i_stabilizer));
    in_stb = cmult(i_v_stab, in); 
    Expression i_h_tm1, i_c_tm1;

    bool has_prev_state = (prev >= 0 || has_initial_state);
    if (prev < 0) {
      if (has_initial_state) {
        // intial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_tm1 = h0[i];
        i_c_tm1 = c0[i];
      }
    } else {  // t > 0
      i_h_tm1 = h[prev][i];
      i_c_tm1 = c[prev][i];
    }

    // input
    Expression i_ait;
    Expression bimb = biases[i][0];
    if (has_prev_state)
        i_ait = affine_transform({ bimb, vars[X2I], in_stb, vars[H2I], i_h_tm1, vars[C2I], i_c_tm1 });
    else
        i_ait = affine_transform({ bimb, vars[X2I], in_stb });
    Expression i_it = logistic(i_ait);
    
    // forget
    // input
#ifdef USE_STANDARD_LSTM_DEFINE
    Expression i_aft;
    Expression bfmb = biases[i][7];
    if (has_prev_state)
        i_aft = affine_transform({ bfmb, vars[X2F], in_stb, vars[H2F], i_h_tm1, vars[C2F], i_c_tm1 });
    else
        i_aft = affine_transform({ bfmb, vars[X2F], in_stb });
    Expression i_ft = logistic(i_aft);
#else
    Expression i_ft = 1.0 - i_it;
#endif

    // write memory cell
    Expression bcmb = biases[i][1];
    Expression i_awt;
    if (has_prev_state)
        i_awt = affine_transform({ bcmb, vars[X2C], in_stb, vars[H2C], i_h_tm1 });
    else
        i_awt = affine_transform({ bcmb, vars[X2C], in_stb });
    Expression i_wt = tanh(i_awt);

    // output
    Expression i_before_add_with_lower_linearly;
    if (has_prev_state) {
      Expression i_nwt = cmult(i_it,i_wt);
      Expression i_crt = cmult(i_ft,i_c_tm1);
      i_before_add_with_lower_linearly = i_crt + i_nwt;
    } else {
      i_before_add_with_lower_linearly = cmult(i_it, i_wt);
    }

    /// add lower layer memory cell
    Expression i_k_t; 
    Expression bkmb = biases[i][3];
    Expression i_k_lowerc = bkmb;
    if (i > 0)
    {
        Expression mc2kmb = biases[i][4];
        i_k_lowerc = i_k_lowerc + cmult(mc2kmb, lower_layer_c); 
    }

    if (has_prev_state)
    {
        Expression q2kmb = biases[i][5];
        i_k_t = logistic(i_k_lowerc + vars[X2K] * in_stb + cmult(q2kmb, i_c_tm1));
    }
    else
        i_k_t = logistic(i_k_lowerc + vars[X2K] * in_stb);
    ct[i] = i_before_add_with_lower_linearly + cmult(i_k_t, (i == 0) ? vars[X2K0] * lower_layer_c : lower_layer_c);


    Expression i_aot;
    Expression bomb = biases[i][2];
    if (has_prev_state)
        i_aot = affine_transform({bomb, vars[X2O], in_stb, vars[H2O], i_h_tm1, vars[C2O], ct[i]});
    else
        i_aot = affine_transform({ bomb, vars[X2O], in_stb });
    Expression i_ot = logistic(i_aot);
    Expression ph_t = tanh(ct[i]);
    in = ht[i] = cmult(i_ot,ph_t);
    lower_layer_c = ct[i];
  }

  if (dropout_rate) return dropout(ht.back(), dropout_rate);
  else return ht.back();
}

// TO DO - Make this correct
// Copied c from the previous step (otherwise c.size()< h.size())
// Also is creating a new step something we want?
// wouldn't overwriting the current one be better?
Expression DGLSTMBuilder::set_h_impl(int prev, const vector<Expression>& h_new) {
  DYNET_ARG_CHECK(h_new.empty() || h_new.size() == layers,
                          "DGLSTMBuilder::set_h expects as many inputs as layers, but got " << h_new.size() << " inputs for " << layers << " layers");
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    Expression h_i = h_new[i];
    Expression c_i = c[t - 1][i];
    h[t][i] = h_i;
    c[t][i] = c_i;
  }
  return h[t].back();
}
// Current implementation : s_new is either {new_c[0],...,new_c[n]}
// or {new_c[0],...,new_c[n],new_h[0],...,new_h[n]}
Expression DGLSTMBuilder::set_s_impl(int prev, const std::vector<Expression>& s_new) {
  DYNET_ARG_CHECK(s_new.size() == layers || s_new.size() == 2 * layers,
                          "DGLSTMBuilder::set_s expects either as many inputs or twice as many inputs as layers, but got " << s_new.size() << " inputs for " << layers << " layers");
  bool only_c = s_new.size() == layers;
  const unsigned t = c.size();
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    Expression h_i = only_c ? h[t - 1][i] : s_new[i + layers];
    Expression c_i = s_new[i];
    h[t][i] = h_i;
    c[t][i] = c_i;
  }
  return h[t].back();
}

void DGLSTMBuilder::copy(const RNNBuilder & rnn) {
  const DGLSTMBuilder & rnn_dglstm = (const DGLSTMBuilder&)rnn;
  DYNET_ARG_CHECK(params.size() == rnn_dglstm.params.size(),
                          "Attempt to copy DGLSTMBuilder with different number of parameters "
                          "(" << params.size() << " != " << rnn_dglstm.params.size() << ")");
  for(size_t i = 0; i < params.size(); ++i)
      for(size_t j = 0; j < params[i].size(); ++j)
        params[i][j] = rnn_dglstm.params[i][j];
}

void DGLSTMBuilder::set_dropout(float d) {
  DYNET_ARG_CHECK(d >= 0.f && d <= 1.f,
                          "dropout rate must be a probability (>=0 and <=1)");
  dropout_rate = d;
}

void DGLSTMBuilder::disable_dropout() {
  dropout_rate = 0.f;
}

ParameterCollection & DGLSTMBuilder::get_parameter_collection() {
  return local_model;
}


} // namespace dynet
