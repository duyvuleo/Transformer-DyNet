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

struct TransformerConfig{
	unsigned _src_vocab_size;
	unsigned _tgt_vocab_size;

	TransformerConfig(const TransformerConfig& tfc){
	}	

	TransformerConfig(){
	}
};

struct TransformerModel {

public:
	explicit TransformerModel(const TransformerConfig& tcf);

	~TransformerModel();

protected:

	DyNetModelPointer _all_params;// all model parameters live in this object pointer. This object will be automatically released once unused!

	LookupParameter _p_svl;// source vocabulary lookup
	LookupParameter _p_tvl;// target vocabulary lookup
};

}; // namespace transformer

