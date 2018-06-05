/* This code is to compute feature scores for moment matching technique
* Developed by Cong Duy Vu Hoang
* Updated: May 2018
*/

#pragma once

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <algorithm>

#include <dynet/dict.h>

void remove_padded_values(WordIdSentence& sent);
void remove_padded_values(WordIdSentence& sent){
	const WordId& pad = sent.back();
	auto iter = std::find(sent.begin(), sent.end(), pad);
	if (iter + 1 != sent.end()) sent.erase(iter + 1, sent.end());
}

using namespace std;
using namespace dynet;

struct MMFeatures{ // abstract class
public:
	unsigned _num_samples;
	unsigned _F_dim;

	explicit MMFeatures(){}
	explicit MMFeatures(unsigned num_samples) : _num_samples(num_samples)
	{}

	virtual ~MMFeatures(){}

	virtual void compute_feature_scores(const WordIdSentences& xs, const WordIdSentences& ys, std::vector<float>& v_scores, unsigned dup=1){
	}

	virtual std::string get_name(){
		return "";
	}
};

// Feature specific for NMT
struct MMFeatures_NMT : public MMFeatures		 
{
	bool _fea_len_ratio = false;
	float _beta = 1.0f;
	
	bool _fea_bi_dict = false;
	std::string _bi_dict_filepath = "";
	float _bi_dict_entry_prob_threshold = 0.5f;
	std::map<std::pair<WordId, WordId>, unsigned> _bi_dict;
		
	bool _fea_pt_smt = false;
	std::string _pt_smt_file_path = "";
	unsigned _ngram = 3; // up to 3-gram matching

	bool _fea_cov = false;

	explicit MMFeatures_NMT(){	
	}

	virtual ~MMFeatures_NMT(){}

	virtual std::string get_name(){
		std::stringstream ss;
		ss << ".n" << this->_num_samples;
		if (_fea_len_ratio)
			ss << ".lr_" << _beta;
		if (_fea_bi_dict)
			ss << ".bd_" << _bi_dict_entry_prob_threshold;
		if (_fea_pt_smt)
			ss << ".psmt";
		if (_fea_cov)
			ss << ".cov";
		
		return ss.str();
	}

	explicit MMFeatures_NMT(unsigned num_samples, 
		bool fea_len_ratio, float beta,
		bool fea_bi_dict, const std::string& bi_dict_filepath, float bi_dict_prob_thresh, dynet::Dict* p_sdict, dynet::Dict* p_tdict,
		bool fea_pt_smt, const std::string& pt_smt_file_path, unsigned ngram,
		bool fea_cov)
		: MMFeatures(num_samples),
		_fea_len_ratio(fea_len_ratio), _beta(beta),
		_fea_bi_dict(fea_bi_dict), _bi_dict_filepath(bi_dict_filepath), _bi_dict_entry_prob_threshold(bi_dict_prob_thresh),
		_fea_pt_smt(fea_pt_smt), _pt_smt_file_path(pt_smt_file_path), _ngram(ngram),
		_fea_cov(fea_cov)
	{			
		if (fea_bi_dict && "" != _bi_dict_filepath){
			ifstream inpf_bi_dict(_bi_dict_filepath);
			assert(inpf_bi_dict);

			std::string line;
			while (std::getline(inpf_bi_dict, line)){
				if ("" == line) continue;

				std::vector<std::string> words = split_words(line);
				assert(words.size() == 3);
				if (atof(words[2].c_str()) >= _bi_dict_entry_prob_threshold && p_sdict->contains(words[0]) && p_tdict->contains(words[1])){
					//cerr << words[0] << "-" << words[1] << "-" << words[2] << endl;
 					WordId src_wid = p_sdict->convert(words[0]);
					WordId trg_wid = p_tdict->convert(words[1]);
					_bi_dict.insert(std::make_pair(std::make_pair(src_wid, trg_wid), _bi_dict.size()));
				}
			}

			cerr << "[MM] - No. of entries in bilingual dictionary: " << _bi_dict.size() << endl;
		}

		this->_F_dim = 0;
		if (_fea_len_ratio) this->_F_dim++;
		if (_fea_bi_dict) this->_F_dim += _bi_dict.size();
		if (_fea_pt_smt){} // FIXME
		if (_fea_cov) this->_F_dim++;
	}

	void get_len_ratio_feature(const WordIdSentence& x, const WordIdSentence& y, float& feature){
		unsigned lx = x.size();
		unsigned ly = y.size();

		// refer to ttp://aclweb.org/anthology/P17-1139
 		if (_beta * lx < (float)ly)
			feature = _beta * (float)lx / ly;
		else feature = (float)ly / (_beta * lx);
			
		// naive feature
		//feature = (float)lx / ly;// simple feature
		
		// length difference
		//feature = std::pow(lx - ly, 2);
	}

	void get_bi_dict_feature(const WordIdSentence& x, const WordIdSentence& y, std::vector<float>& features){
		features.clear();
		features.resize(_bi_dict.size(), 0.f);
		for (auto& sword : x){
			for (auto& tword : y){
				//cerr << sword << "-" << tword << endl;
				const auto& iter = _bi_dict.find(std::make_pair(sword, tword));// FIXME: is this slow?
				if (iter != _bi_dict.end())//found
					features[iter->second] = 1.f;				
			}
		}
	}

	void get_pt_smt_feature(const WordIdSentence& x, const WordIdSentence& y, std::vector<float>& features){
		// FIXME
	}

	void get_cov_feature(const WordIdSentence& x, const WordIdSentence& y, const vector<float>& align_probs, float& feature){
		// FIXME
	}
	
	virtual void compute_feature_scores(const WordIdSentences& xs, const WordIdSentences& ys, std::vector<float>& v_scores, unsigned dup=1){
		v_scores.clear();
		
		for (unsigned s = 0; s < xs.size(); s++){
			const auto& src = xs[s];
			auto sample = ys[s];
			remove_padded_values(sample);
			/*cerr << "src: ";
			for (auto& w : src) cerr << w << " ";
			cerr << endl;
			cerr << "sample: ";
                        for (auto& w : sample) cerr << w << " ";
                        cerr << endl;*/
			
			if (_fea_len_ratio){
				float f = 0.f;
				this->get_len_ratio_feature(src, sample, f);
				v_scores.push_back(f);
			}
			
			if (_fea_bi_dict){
				std::vector<float> v_f;
				this->get_bi_dict_feature(src, sample, v_f);
				v_scores.insert(v_scores.end(), v_f.begin(), v_f.end());
			}
			
			if (_fea_pt_smt){
				// FIXME
			}

			if (_fea_cov){
				// FIXME
			}
			
	                if (dup > 1){
				// naive version
				unsigned old_size = v_scores.size();				
				for (unsigned d = 1; d < dup; d++){
					for (unsigned i = old_size - this->_F_dim; i < old_size; i++)
						v_scores.push_back(v_scores[i]);
				}				
			}		
		}
	}
};

// Feature specific for DP (dependency parsing)
struct MMFeatures_DP : public MMFeatures		       
{
	explicit MMFeatures_DP(){	
	}

	explicit MMFeatures_DP(unsigned num_samples)
		: MMFeatures(num_samples)
	{	
	}

	virtual ~MMFeatures_DP(){}
};

// Feature specific for CP (constituency parsing)
struct MMFeatures_CP : public MMFeatures		 
{
	explicit MMFeatures_CP(){	
	}

	explicit MMFeatures_CP(unsigned num_samples)
		: MMFeatures(num_samples)
	{	
	}

	virtual ~MMFeatures_CP(){}
};

// Feature specific for WO (word ordering)
struct MMFeatures_WO : public MMFeatures		 
{
	std::vector<bool> _ablation = {true, true, true};

	explicit MMFeatures_WO(){	
	}

	explicit MMFeatures_WO(unsigned num_samples, bool do_len_ratio=true, bool do_precision=true, bool do_recall=true)
		: MMFeatures(num_samples)
	{			
                _ablation[0] = do_len_ratio;
		_ablation[1] = do_precision;
		_ablation[2] = do_recall;
		this->_F_dim = 0;
		for (auto b : _ablation)
			if (b) this->_F_dim++;
	}

	virtual std::string get_name(){
		std::stringstream ss;
		ss << ".n" << this->_num_samples;
		if (_ablation[0])
			ss << ".lr";
		if (_ablation[1])
                        ss << ".p";
		if (_ablation[2])
                        ss << ".r";
		
		return ss.str();
	}

	virtual void compute_feature_scores(const WordIdSentences& xs, const WordIdSentences& ys, std::vector<float>& v_scores, unsigned dup=1){
		v_scores.clear();
		
		for (unsigned s = 0; s < xs.size(); s++){
			auto src = xs[s];
			auto sample = ys[s];
			remove_padded_values(sample);
			/*cerr << "src: ";
			for (auto& w : src) cerr << w << " ";
			cerr << endl;
			cerr << "sample: ";
                        for (auto& w : sample) cerr << w << " ";
                        cerr << endl;*/

			// get sentence lengths
			unsigned lx = src.size() - 2;// not including <s> and </s>
			unsigned ly = sample.size() - 2;
			//cerr << "lx=" << lx << endl;
			//cerr << "ly=" << ly << endl;
			if (ly == 0){// bad sample: <s> </s> (or empty)
				cerr << "src: ";
                        	for (auto& w : src) cerr << w << " ";
                        	cerr << endl;
                        	cerr << "sample: ";
                        	for (auto& w : sample) cerr << w << " ";
                        	cerr << endl;
			}
			
			// constraint 1: length ratio
			if (_ablation[0]){
				if (lx < ly) v_scores.push_back((float)lx / ly);
				else v_scores.push_back((float)ly / lx);
			}

			if (_ablation[1] || _ablation[2]){
			        WordIdSentence intersection;
				std::sort(src.begin(), src.end());
				std::sort(sample.begin(), sample.end());
				std::set_intersection(src.begin(), src.end(), sample.begin(), sample.end(), std::back_inserter(intersection));

				// constraint 2: precision (refers to no. of words in source appear in sample)
				if (_ablation[1]){
					float precision = (float)(intersection.size() - 2) / ly;			
					v_scores.push_back(precision);
				}
	
				// constraint 3: recall (refers to no. of words in sample appear in source)
				if (_ablation[2]){
					float recall = (float)(intersection.size() - 2) / lx;			
					v_scores.push_back(recall);
				}
			}
			 
			// combine?: (|x| - |y|)^2 + ( #{w \in x & w \in y for \all w} - |x|)^2		
			
                       if (dup > 1){
				// naive version
				unsigned old_size = v_scores.size();				
				for (unsigned d = 1; d < dup; d++){
					for (unsigned i = old_size - this->_F_dim; i < old_size; i++)
						v_scores.push_back(v_scores[i]);
				}				
			}
		}
	}

	virtual ~MMFeatures_WO(){}
};


