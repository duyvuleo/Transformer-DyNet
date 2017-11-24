#pragma once

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"

using namespace dynet;

dynet::Expression arange(dynet::ComputationGraph &cg, unsigned begin, unsigned end, bool log_transform, std::vector<float> *aux_mem) 
{
	aux_mem->clear();
	for (unsigned i = begin; i < end; ++i) 
		aux_mem->push_back((log_transform) ? log(1.0 + i) : i);
	return input(cg, dynet::Dim({end-begin}), aux_mem, dynet::default_device);
}

dynet::Expression repeat(dynet::ComputationGraph &cg, unsigned num, float value, std::vector<float> *aux_mem) 
{
	aux_mem->clear();
	aux_mem->resize(num, value);
	return input(cg, dynet::Dim({num}), aux_mem, dynet::default_device);
}

dynet::Expression dither(dynet::ComputationGraph &cg, const dynet::Expression &expr, float pad_value, std::vector<float> *aux_mem)
{
	const auto& shape = cg.nodes[expr.i]->dim;
	aux_mem->clear();
	aux_mem->resize(shape.cols(), pad_value);
	dynet::Expression padding = input(cg, dynet::Dim({shape.cols()}), aux_mem, dynet::default_device);
	dynet::Expression padded = dynet::concatenate(std::vector<dynet::Expression>({padding, expr, padding}));
	dynet::Expression left_shift = dynet::pickrange(padded, 2, shape.rows() + 2);
	dynet::Expression right_shift = dynet::pickrange(padded, 0, shape.rows());
	return dynet::concatenate_cols(std::vector<dynet::Expression>({left_shift, expr, right_shift}));
}

// binary boolean functions
dynet::Expression eq(const dynet::Expression &expr, float value, float epsilon=0.1) 
{
	return min(dynet::rectify(expr - (value - epsilon)), dynet::rectify(-expr + (value + epsilon))) / epsilon; 
}

dynet::Expression geq(const dynet::Expression &expr, float value, dynet::Expression &one, float epsilon=0.01) 
{
	return min(one, dynet::rectify(expr - (value - epsilon)) / epsilon);
}

dynet::Expression leq(const dynet::Expression &expr, float value, dynet::Expression &one, float epsilon=0.01) 
{
	return min(one, dynet::rectify((value + epsilon) - expr) / epsilon);
}

// @Vu -- this should be implemented in dynet!
dynet::Expression softplus(const dynet::Expression &expr) 
{
	return dynet::log(dynet::exp(expr) + 1);// https://www.tensorflow.org/api_docs/python/tf/nn/softplus
}

// @Vu: this layer_norm_colwise is an upgrade of dynet::layer_norm which only supports vector.
// Here, x can be either vector or matrix.
// refer to https://github.com/clab/dynet/issues/1066
/*dynet::Expression layer_norm_colwise(const dynet::Expression& x, const dynet::Expression& g, const dynet::Expression& b, float epsilon=1e-8){
	dynet::Expression mu = dynet::transpose(dynet::mean_dim(x, {0}));
	mu = dynet::concatenate(std::vector<dynet::Expression>(x.dim()[0], mu));

	dynet::Expression sigma = dynet::transpose(dynet::std_dim(x, {0}));
	sigma = dynet::concatenate(std::vector<dynet::Expression>(x.dim()[0], sigma));

	dynet::Expression x_centered = x - mu;

	return dynet::cmult(g, dynet::cdiv(x_centered, sigma + epsilon)) + b;
}*/ // version 2: a bit faster

Expression layer_norm_colwise(const Expression& x, const Expression& g, const Expression& b){
	std::vector<Expression> vCols(x.dim().d[1]);
	for (unsigned i = 0; i < x.dim().d[1]; i++){ 
		dynet::Expression c_x = dynet::select_cols(x, {i});
		vCols[i] = dynet::layer_norm(c_x, g, b);
	}
	return dynet::concatenate_cols(vCols);
}// version 1


