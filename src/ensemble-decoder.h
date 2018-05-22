/* This is an implementation of ensemble decoder for Transformer, adapted from ensemble-decoder.{h,cc} of lamtram toolkit (https://github.com/neubig/lamtram).
 * Adapted by Cong Duy Vu Hoang (duyvuleo@gmail.com, vhoang2@student.unimelb.edu.au) 
 */

#pragma once

#include <dynet/tensor.h>
#include <dynet/dynet.h>

#include <vector>
#include <cfloat>

#include "transformer.h"

using namespace std;
using namespace dynet;
using namespace transformer;

//#define USE_BEAM_SEARCH_LENGTH_NORMALISATION
#define USE_BEAM_SEARCH_LENGTH_NORMALISATION_NEMATUS
//#define USE_BEAM_SEARCH_LENGTH_NORMALISATION_GNMT
float _len_norm_alpha = 0.6f;// global variable

class EnsembleDecoderHyp {
public:
	EnsembleDecoderHyp(float score, const WordIdSentence & sent, const WordIdSentence & align) :
		_score(score), _sent(sent), _align(align) { }

	float get_score() const { return _score; }
	float get_norm_score() const { 
#if defined(USE_BEAM_SEARCH_LENGTH_NORMALISATION)
		return _score / _sent.size(); 
#elif defined(USE_BEAM_SEARCH_LENGTH_NORMALISATION_NEMATUS)
		return _score / std::pow(_sent.size(), _len_norm_alpha); 
#elif defined(USE_BEAM_SEARCH_LENGTH_NORMALISATION_GNMT)
		return _score / std::pow((5.f + _sent.size()) / 6.f , _len_norm_alpha); 
#else
		return _score;
#endif
	}

	const WordIdSentence & get_sentence() const { return _sent; }
	const WordIdSentence & get_alignment() const { return _align; }

protected:

	float _score;
	WordIdSentence _sent;
	WordIdSentence _align;
};

class BatchedEnsembleDecoderHyp {
public:
	BatchedEnsembleDecoderHyp(const std::vector<float>& scores, const WordIdSentences & sents, const WordIdSentences & aligns) :
		_scores(scores), _sents(sents), _aligns(aligns) { }

	std::vector<float>& get_scores() { return _scores; }
	std::vector<float> get_norm_scores() const {
		std::vector<float> norm_scores;
		unsigned i = 0;
		for (const float& score : _scores){
#if defined(USE_BEAM_SEARCH_LENGTH_NORMALISATION)
			norm_scores.push_back(score / _sents[i++].size()); 
#elif defined(USE_BEAM_SEARCH_LENGTH_NORMALISATION_NEMATUS)
			norm_scores.push_back(score / std::pow(_sents[i++].size(), _len_norm_alpha)); 
#elif defined(USE_BEAM_SEARCH_LENGTH_NORMALISATION_GNMT)
			norm_scores.push_back(score / std::pow((5.f + _sents[i++].size()) / 6.f , _len_norm_alpha)); 
#else
			norm_scores.push_back(score);
#endif
		}
		
		return norm_scores;
	}

	const WordIdSentences & get_sentences() const { return _sents; }
	const WordIdSentences & get_alignments() const { return _aligns; }

protected:

	std::vector<float> _scores;
	WordIdSentences _sents;
	WordIdSentences _aligns;
};

typedef std::shared_ptr<EnsembleDecoderHyp> EnsembleDecoderHypPtr;
typedef std::shared_ptr<BatchedEnsembleDecoderHyp> BatchedEnsembleDecoderHypPtr;

inline bool operator<(const EnsembleDecoderHypPtr & lhs, const EnsembleDecoderHypPtr & rhs) {
#if defined(USE_BEAM_SEARCH_LENGTH_NORMALISATION)
	//score with word-based length normalization.
	float score_l = lhs->get_score()/lhs->get_sentence().size(), score_r = rhs->get_score()/rhs->get_sentence().size();
	if( score_l != score_r) return score_l > score_r;
	return lhs->get_sentence() < rhs->get_sentence();
#elif defined(USE_BEAM_SEARCH_LENGTH_NORMALISATION_NEMATUS)
	//score with word-based length normalization using Nematus style (Ly^alpha) (https://arxiv.org/abs/1703.04357)
	float score_l = lhs->get_score() / std::pow(lhs->get_sentence().size(), _len_norm_alpha), score_r = rhs->get_score() / std::pow(rhs->get_sentence().size(), _len_norm_alpha);
	if( score_l != score_r) return score_l > score_r;
	return lhs->get_sentence() < rhs->get_sentence();
#elif defined(USE_BEAM_SEARCH_LENGTH_NORMALISATION_GNMT)
	//score with word-based length normalization using GNMT style ((5 + L)^alpha / 6^alpha) (https://arxiv.org/pdf/1609.08144.pdf)
	// ToDo (FIXME): add alignment score to include a coverage penalty to favor translations that fully cover the source sentence according to the attention matrix.
	float score_l = lhs->get_score() / std::pow((5.f + lhs->get_sentence().size()) / 6.f, _len_norm_alpha), score_r = rhs->get_score() / std::pow((5.f + rhs->get_sentence().size()) / 6.f, _len_norm_alpha);
	if( score_l != score_r) return score_l > score_r;
	return lhs->get_sentence() < rhs->get_sentence();
#else
	if(lhs->get_score() != rhs->get_score()) return lhs->get_score() > rhs->get_score();
	return lhs->get_sentence() < rhs->get_sentence();
#endif
}

typedef tuple<float,int,int,int> Beam_Info;

class EnsembleDecoder {
public:
	EnsembleDecoder(dynet::Dict& td);
	~EnsembleDecoder() {}

	// --- non-batched version
	EnsembleDecoderHypPtr generate(dynet::ComputationGraph& cg
		, const WordIdSentence & sent_src
		, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models);
	std::vector<EnsembleDecoderHypPtr> generate_nbest(dynet::ComputationGraph& cg
		, const WordIdSentence & sent_src
		, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
		, unsigned nbest_size);
	// ---
	// --- batched version
	BatchedEnsembleDecoderHypPtr generate(dynet::ComputationGraph& cg
		, const WordIdSentences &sent_src_minibatch
		, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models);
	std::vector<BatchedEnsembleDecoderHypPtr> generate_nbest(dynet::ComputationGraph& cg
		, const WordIdSentences &sent_src_batch
		, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
		, unsigned nbest_size);
	// ---

	// Ensemble together probabilities or log probabilities for a single word
	Expression ensemble_probs(const std::vector<Expression> & in, dynet::ComputationGraph & cg);
	Expression ensemble_logprobs(const std::vector<Expression> & in, dynet::ComputationGraph & cg);
	
	float get_word_pen() const { return _word_pen; }
	float get_unk_pen() const { return _unk_pen; }
	std::string get_ensemble_operation() const { return _ensemble_operation; }
	void set_word_pen(float word_pen) { _word_pen = word_pen; }
	void set_unk_pen(float unk_pen) { _unk_pen = unk_pen; }
	void set_ensemble_operation(const std::string & ensemble_operation) { _ensemble_operation = ensemble_operation; }

	int get_beam_size() const { return _beam_size; }
	void set_beam_size(unsigned beam_size) { _beam_size = beam_size; }
	int get_size_limit() const { return _size_limit; }
	void set_size_limit(unsigned size_limit) { _size_limit = size_limit; }
	void set_length_ratio(unsigned length_ratio){ _length_ratio = length_ratio; }

protected:

	float _word_pen;
	float _unk_pen, _unk_log_prob;
	WordId _unk_id;
	unsigned _size_limit;
	unsigned _beam_size;
	unsigned _length_ratio;
	std::string _ensemble_operation;

	bool _verbose;
};

EnsembleDecoder::EnsembleDecoder(dynet::Dict& td)
	: _word_pen(0.f), _unk_pen(0.f), _size_limit(500), _beam_size(1), _length_ratio(0), _ensemble_operation("sum"), _verbose(false) 
{
	_unk_id = td.convert("<unk>");
	_unk_log_prob = -std::log(td.size());// penalty score for <unk>
}

dynet::Expression EnsembleDecoder::ensemble_probs(const std::vector<dynet::Expression> & v_ins, dynet::ComputationGraph & cg) {
	if(v_ins.size() == 1) return v_ins[0];
	return dynet::average(v_ins);
}

dynet::Expression EnsembleDecoder::ensemble_logprobs(const std::vector<dynet::Expression> & v_ins, dynet::ComputationGraph & cg) {
	if(v_ins.size() == 1) return v_ins[0];
	dynet::Expression i_average = dynet::average(v_ins);
	return dynet::log_softmax({i_average});
}

inline int max_len(const WordIdSentence & sent) { return sent.size(); }
inline int max_len(const std::vector<WordIdSentence> & sent) {
	size_t val = 0;
	for (const auto & s : sent){
		val = std::max(val, s.size()); 
	}
	return val;
}

inline int get_word(const std::vector<WordIdSentence> & vec, int t) { return vec[0][t]; }
inline int get_word(const WordIdSentence & vec, int t) { return vec[t]; }

EnsembleDecoderHypPtr EnsembleDecoder::generate(dynet::ComputationGraph& cg
	, const WordIdSentence & sent_src
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models) 
{
	auto nbest = generate_nbest(cg, sent_src, v_models, 1);
	return (nbest.size() > 0 ? nbest[0] : EnsembleDecoderHypPtr());
}

std::vector<EnsembleDecoderHypPtr> EnsembleDecoder::generate_nbest(dynet::ComputationGraph& cg
	, const WordIdSentence & sent_src
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
	, unsigned nbest_size) 
{ 
	// sentinel symbols
	const transformer::SentinelMarkers& sm = v_models[0].get()->get_config()._sm;
	
	// compute source representation
	std::vector<dynet::Expression> v_src_reps;
	for (auto & tf : v_models){
		v_src_reps.push_back(tf.get()->compute_source_rep(cg, WordIdSentences(1, sent_src)/*pseudo batch (1)*/));
	}

	// the n-best hypotheses
	std::vector<EnsembleDecoderHypPtr> nbest;

	// create the initial hypothesis
	std::vector<EnsembleDecoderHypPtr> curr_beam(1, EnsembleDecoderHypPtr(new EnsembleDecoderHyp(0.0, WordIdSentence(1, sm._kTGT_SOS), WordIdSentence(1, 0))));

	int bid;

	// limit the output length
	if (_length_ratio > 0)
		_size_limit = sent_src.size() * _length_ratio;// not generating target with the approximate length "_length_ratio times" than the source length

	// perform decoding
	for (unsigned sent_len = 0; sent_len <= _size_limit; sent_len++) {
		// this vector will hold the best IDs
		std::vector<Beam_Info> next_beam_id(_beam_size+1, Beam_Info(-DBL_MAX,-1,-1,-1));

		// go through all the hypothesis IDs
		for (int hypid = 0; hypid < (int)curr_beam.size(); hypid++) {
			EnsembleDecoderHypPtr curr_hyp = curr_beam[hypid];
			const WordIdSentence& sent = curr_hyp->get_sentence();// partial generated sentence from current hypo in the beam

			if (sent_len != 0 && *sent.rbegin() == sm._kTGT_EOS) continue;

			cg.checkpoint();

			// perform the forward step on all models
			std::vector<dynet::Expression> i_softmaxes, i_aligns;
			for(int j : boost::irange(0, (int)v_models.size())){
				i_softmaxes.push_back(v_models[j].get()->step_forward(cg, v_src_reps[j]
					, sent
					, _ensemble_operation == "logsum"
					, i_aligns));
			}

			// ensemble and calculate the likelihood
			dynet::Expression i_softmax, i_logprob;
			if (_ensemble_operation == "sum") {
				i_softmax = ensemble_probs(i_softmaxes, cg);
				i_logprob = dynet::log({i_softmax});
			}
			else if (_ensemble_operation == "logsum") {
				i_logprob = ensemble_logprobs(i_softmaxes, cg);
			}
			else
				assert(string("Bad ensembling operation: " + _ensemble_operation).c_str());

			// get the (log) softmax predictions
			std::vector<float> softmaxes = dynet::as_vector(cg.incremental_forward(i_logprob));

			// add the word penalty
			if (_word_pen != 0.f) {
				for(size_t i = 0; i < softmaxes.size(); i++)
					softmaxes[i] += _word_pen;
			}
			
			// set up unk penalty
			if (_unk_id >= 0) softmaxes[_unk_id] += _unk_pen * _unk_log_prob;

			// find the best aligned source, if any alignments exists
			WordId best_align = -1;
			if (i_aligns.size() != 0) {
				dynet::Expression ens_align = dynet::sum(i_aligns);
				std::vector<float> align = dynet::as_vector(cg.incremental_forward(ens_align));
				best_align = 0;
				for(size_t aid = 0; aid < align.size(); aid++)
					if(align[aid] > align[best_align])
						best_align = aid;
			}

			// find the best IDs in the beam
			for (int wid = 0; wid < (int)softmaxes.size(); wid++) {
				float my_score = curr_hyp->get_score() + softmaxes[wid];
				for (bid = _beam_size; bid > 0 && my_score > std::get<0>(next_beam_id[bid-1]); bid--)
					next_beam_id[bid] = next_beam_id[bid-1];
				next_beam_id[bid] = Beam_Info(my_score, hypid, wid, best_align);
			}

			cg.revert();
		}

		// create the new hypotheses
		std::vector<EnsembleDecoderHypPtr> next_beam;
		for (unsigned i = 0; i < _beam_size; i++) {
			float score = std::get<0>(next_beam_id[i]);
			int hypid = std::get<1>(next_beam_id[i]);
			int wid = std::get<2>(next_beam_id[i]);
			int aid = std::get<3>(next_beam_id[i]);

			if (hypid == -1) break;// never happens?

			WordIdSentence next_sent = curr_beam[hypid]->get_sentence();
			next_sent.push_back(wid);

			WordIdSentence next_align = curr_beam[hypid]->get_alignment();
			next_align.push_back(aid);

			EnsembleDecoderHypPtr hyp(new EnsembleDecoderHyp(score, next_sent, next_align));

			if (wid == sm._kTGT_EOS && hyp->get_sentence().size() == 2) //excluding empty generation, e.g., <s> </s>
				continue;

			if (wid == sm._kTGT_EOS || sent_len == _size_limit)
				nbest.push_back(hyp);

			next_beam.push_back(hyp);
		}

		curr_beam = next_beam;

		// check if we're done with search
		if(nbest.size() != 0) {
			std::sort(nbest.begin(), nbest.end());

			if(nbest.size() > nbest_size) nbest.resize(nbest_size);
			if(nbest.size() == nbest_size && (curr_beam.size() == 0 || (*nbest.rbegin())->get_norm_score() >= next_beam[0]->get_norm_score()))
				return nbest;
		}

		// if current beam size is 0, stop!
		if(curr_beam.size() == 0) break;
	}

	cg.clear();// maybe trying to release memory!

	if (_verbose) cerr << "WARNING: Generated sentence size exceeded " << _size_limit << ". Truncating." << endl;

	return nbest;
}

// -----------------------------------------------------------------------------------
// WIP: to support batch decoding
BatchedEnsembleDecoderHypPtr EnsembleDecoder::generate(dynet::ComputationGraph& cg
	, const WordIdSentences &sent_src_minibatch
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models) 
{
	auto nbest = generate_nbest(cg, sent_src_minibatch, v_models, 1);
	return (nbest.size() > 0 ? nbest[0] : BatchedEnsembleDecoderHypPtr());// FIXME: check nbest[0]
}

std::vector<BatchedEnsembleDecoderHypPtr> EnsembleDecoder::generate_nbest(dynet::ComputationGraph& cg
	, const WordIdSentences &sent_src_batch
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
	, unsigned nbest_size) 
{ 
	// minibatch size
	unsigned bsize = sent_src_batch.size();

	// target vocab size
	const unsigned target_vocab_size =  v_models[0].get()->get_target_dict().size();

	// sentinel symbols
	const transformer::SentinelMarkers& sm = v_models[0].get()->get_config()._sm;
	  
	// compute source representation
	std::vector<dynet::Expression> v_src_reps;// batched
	for (auto & tf : v_models){
		v_src_reps.push_back(tf.get()->compute_source_rep(cg, sent_src_batch));
	}

	// the n-best hypotheses
	std::vector<BatchedEnsembleDecoderHypPtr> nbest;

	// create the initial hypothesis
	//std::vector<BatchedEnsembleDecoderHypPtr> curr_beam(1, BatchedEnsembleDecoderHypPtr(new BatchedEnsembleDecoderHyp(0.0, WordIdSentences(1, WordIdSentence(1, sm._kTGT_SOS)), WordIdSentences(1, WordIdSentence(1, 0)))));

	int bid;

	// limit the output length
	size_t max_len = sent_src_batch[0].size();
	for(size_t i = 1; i < bsize; i++) max_len = std::max(max_len, sent_src_batch[i].size());
	if (_length_ratio > 0)
		_size_limit = max_len * _length_ratio;// not generating target with the approximate length "TARGET_LENGTH_LIMIT_FACTOR times" than the source length

	// perform decoding
	for (unsigned sent_len = 0; sent_len <= _size_limit; sent_len++) {
		// this vector will hold the best IDs
		std::vector<Beam_Info> next_beam_id(_beam_size+1, Beam_Info(-DBL_MAX,-1,-1,-1));

		// go through all the hypothesis IDs
		//for (int hypid = 0; hypid < (int)curr_beam.size(); hypid++) 
		{
			/*
			BatchedEnsembleDecoderHypPtr curr_hyp = curr_beam[hypid];
			const WordIdSentences& sents = curr_hyp->get_sentences();// partial generated sentences from current hypo in the beam

			if (sent_len != 0 && *sents[0].rbegin() == sm._kTGT_EOS) continue;

			cg.checkpoint();

			// perform the forward step on all models
			std::vector<dynet::Expression> i_softmaxes, i_aligns;// batched
			for(int j : boost::irange(0, (int)v_models.size())){
				i_softmaxes.push_back(v_models[j].get()->step_forward(cg, v_src_reps[j]
					, sents
					, _ensemble_operation == "logsum"
					, i_aligns));
			}

			// ensemble and calculate the likelihood
			dynet::Expression i_softmax, i_logprob;// batched
			if (_ensemble_operation == "sum") {
				i_softmax = ensemble_probs(i_softmaxes, cg);
				i_logprob = dynet::log({i_softmax});
			}
			else if (_ensemble_operation == "logsum") {
				i_logprob = ensemble_logprobs(i_softmaxes, cg);
			}
			else
				assert(string("Bad ensembling operation: " + _ensemble_operation).c_str());

			// get the (log) softmax predictions
			std::vector<float> softmaxes = dynet::as_vector(cg.incremental_forward(i_logprob));// batch_size x TARGET_VOCAB_SIZE

			// add the word penalty
			if (_word_pen != 0.f) {
				for(size_t i = 0; i < softmaxes.size(); i++)
					softmaxes[i] += _word_pen;
			}
			
			// set up unk penalty
			if (_unk_id >= 0){
				for (size_t i = 0; i < bsize; i++)
					softmaxes[_unk_id + i * target_vocab_size] += _unk_pen * _unk_log_prob;
			}

			// find the best aligned source, if any alignments exists
			// FIXME: need to process a batch of alignment matrices
			// ...
			//

			// find the best IDs in the beam
			// FIXME: batched?
			for (int wid = 0; wid < (int)softmaxes.size(); wid++) {
				float my_score = curr_hyp->get_scores() + softmaxes[wid];
				for (bid = _beam_size; bid > 0 && my_score > std::get<0>(next_beam_id[bid-1]); bid--)
					next_beam_id[bid] = next_beam_id[bid-1];
				next_beam_id[bid] = Beam_Info(my_score, hypid, wid, best_align);
			}
			*/
			// FIXME
		}

	}

	cg.clear();// maybe trying to release memory!

	if (_verbose) cerr << "WARNING: Generated sentence size exceeded " << _size_limit << ". Truncating." << endl;

	return nbest;
}

// batch decoding
// ToDo (FIXME): 

