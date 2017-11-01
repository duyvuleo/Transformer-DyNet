/**
 * \file dglstm.h
 * \brief Helper structures to build recurrent units
 *
 * \details TODO: Create documentation and explain rnns, etc...
 */
#ifndef DYNET_DGLSTM_H_
#define DYNET_DGLSTM_H_

#include "dynet/dynet.h"
#include "dynet/rnn.h"
#include "dynet/expr.h"

namespace dynet {

class ParameterCollection;
/**
 * \ingroup rnnbuilders
 * \brief DGLSTMBuilder creates an DGLSTM unit with coupled input and forget gate as well as peepholes connections.
 *
 * \details Depth-Gated Long Short Term Memory (DGLSTM) (Kaisheng et al., 2015; https://arxiv.org/abs/1508.03790)
 *
 */
struct DGLSTMBuilder : public RNNBuilder {
  /**
   * \brief Default constructor
   */
  DGLSTMBuilder() = default;
  /**
   * \brief Constructor for the DGLSTMBuilder
   *
   * \param layers Number of layers
   * \param input_dim Dimention of the input \f$x_t\f$
   * \param hidden_dim Dimention of the hidden states \f$h_t\f$ and \f$c_t\f$
   * \param model ParameterCollection holding the parameters
   */
  explicit DGLSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       ParameterCollection& model);

  Expression back() const override { return (cur == -1? h0.back() : h[cur].back()); }

  std::vector<Expression> final_h() const override { return (h.size() == 0 ? h0 : h.back()); }
  std::vector<Expression> final_s() const override {
    std::vector<Expression> ret = (c.size() == 0 ? c0 : c.back());
    for(auto my_h : final_h()) ret.push_back(my_h);
    return ret;
  }

  /**
   * @brief Number of components in `h_0`
   * @details For `LSTMBuilder`, this corresponds to `2 * layers` because it includes the initial cell state \f$c_0\f$
   * @return `2 * layers`
   */
  unsigned num_h0_components() const override { return 2 * layers; }

  std::vector<Expression> get_h(RNNPointer i) const override { return (i == -1 ? h0 : h[i]); }
  /**
   * @brief Get the final state of the hidden layer
   * @details For `LSTMBuilder`, this consists of a vector of the memory cell values for each layer (l1, l2, l3),
   *          followed by the hidden state values
   * @return {c_{l1}, c_{l1}, ..., h_{l1}, h_{l2}, ...}
   */
  std::vector<Expression> get_s(RNNPointer i) const override {
    std::vector<Expression> ret = (i == -1 ? c0 : c[i]);
    for(auto my_h : get_h(i)) ret.push_back(my_h);
    return ret;
  }

  void copy(const RNNBuilder & params) override;

  /**
   * \brief Set the dropout rates to a unique value
   * \details This has the same effect as `set_dropout(d,d_h,d_c)` except that all the dropout rates are set to the same value.
   * \param d Dropout rate to be applied on all of \f$x,h,c\f$
   */
  void set_dropout(float d);
  /**
   * \brief Set all dropout rates to 0
   * \details This is equivalent to `set_dropout(0)` or `set_dropout(0,0,0)`
   *
   */
  void disable_dropout();
  /**
   * \brief Get parameters in LSTMBuilder
   */
  ParameterCollection & get_parameter_collection() override;

protected:
  void new_graph_impl(ComputationGraph& cg, bool update) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
  Expression add_input_impl(int prev, const Expression& x) override;
  Expression set_h_impl(int prev, const std::vector<Expression>& h_new) override;
  Expression set_s_impl(int prev, const std::vector<Expression>& s_new) override;

private:
  void initialize_biases(); 

public:
  ParameterCollection local_model;

  // first index is layer, then ...
  std::vector<std::vector<Parameter>> params;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h, c;

  std::vector<std::vector<Expression>> biases;

  std::vector<unsigned> input_dims;  /// input dimension at each layer

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;

  unsigned layers;
  unsigned hid = 0;

private:
  ComputationGraph  *_cg;

};

} // namespace dynet

#endif
