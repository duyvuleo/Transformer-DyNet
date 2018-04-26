/*
 * This is an implementation of the following work:
 * Dual Learning for Machine Translation
 * Yingce Xia, Di He, Tao Qin, Liwei Wang, Nenghai Yu, Tie-Yan Liu, Wei-Ying Ma
 * https://papers.nips.cc/paper/6469-dual-learning-for-machine-translation.pdf
 * Developed by Cong Duy Vu Hoang (vhoang2@student.unimelb.edu.au)
 * Date: April 2018
 *
*/

// We call this framework as "round tripping" instead of "dual learning". 

#include "../transformer.h" // transformer
#include "../transformer-lm.h" // transformer-based lm

#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace dynet;
using namespace transformer;
using namespace boost::program_options;

unsigned MAX_EPOCH = 10;
unsigned DEV_ROUND = 25000;

bool VERBOSE;

int main_body(variables_map vm);

typedef WordIdSentences MonoData;

std::string get_sentence(const WordIdSentence& source, Dict& td);

// create SGD trainer
dynet::Trainer* create_sgd_trainer(unsigned opt_type, float lr, dynet::ParameterCollection& model);

// read the data
MonoData read_mono_data(dynet::Dict& d, const string &filepath, unsigned max_seq_len=0, unsigned load_percent=100);

// main round tripping function
void run_round_tripping(std::vector<std::shared_ptr<transformer::TransformerModel>>& v_tm_models, std::vector<std::shared_ptr<transformer::TransformerLModel>>& v_alm_models
		, const MonoData& mono_s, const MonoData& mono_t
		, const WordIdCorpus& dev_cor // for evaluation
		, unsigned K, unsigned beam_size, float alpha, float gamma_1, float gamma_2 /*hyper-parameters of round tripping framework*/
		, unsigned opt_type);

int main(int argc, char** argv) {
	dynet::initialize(argc, argv);

	// command line processing
	variables_map vm; 
	options_description opts("Allowed options");
	opts.add_options()
		("help", "print help message")
		("config,c", value<string>(), "config file specifying additional command line options")
		//-----------------------------------------		
		("model-path-s2t", value<std::string>()->default_value("."), "pre-trained path for the source-to-target transformer model will be loaded from this folder")
		("model-path-t2s", value<std::string>()->default_value("."), "pre-trained path for the target-to-source transformer model will be loaded from this folder")
		("model-path-s", value<std::string>()->default_value("."), "pre-trained path for the target language model will be loaded from this folder")
		("model-path-t", value<std::string>()->default_value("."), "pre-trained path for the source language model will be loaded from this folder")
		("devel,d", value<string>(), "file containing development parallel sentences, with "
			"each line consisting of source ||| target.")
		("max-seq-len", value<unsigned>()->default_value(0), "limit the sequence length loaded from mono data; none by default")
		("mono-load-percent", value<unsigned>()->default_value(100), "limit the use of <num> percent of sequences in monolingual data; full by default")
		//-----------------------------------------	
		("K", value<unsigned>()->default_value(10), "the K value for sampling K-best translations from transformer models")
		("beam_size", value<unsigned>()->default_value(5), "the beam size of beam search decoding")
		("alpha", value<float>()->default_value(0.5f), "the alpha hyper-parameter for balancing the rewards")
		("gamma_1", value<float>()->default_value(0.1f), "the gamma 1 hyper-parameter for stochastic gradient update in tuning source-to-target transformer model")
		("gamma_2", value<float>()->default_value(0.1f), "the gamma 2 hyper-parameter for stochastic gradient update in tuning target-to-source transformer model")
		//-----------------------------------------
		("mono_s", value<string>()->default_value(""), "File to read the monolingual source from")
		("mono_t", value<string>()->default_value(""), "File to read the monolingual target from")
		//-----------------------------------------
		("epoch,e", value<unsigned>()->default_value(50), "number of training epochs, 50 by default")
		("dev_round", value<unsigned>()->default_value(25000), "number of rounds for evaluating over development data, 25000 by default")
		//-----------------------------------------
		("verbose,v", "be extremely chatty")
	;
	store(parse_command_line(argc, argv, opts), vm); 
	if (vm.count("config") > 0)
	{
		ifstream config(vm["config"].as<string>().c_str());
		store(parse_config_file(config, opts), vm); 
	}
	notify(vm);
	
	if (vm.count("help")) {// FIXME: check the missing ones?
		cout << opts << "\n";
		return EXIT_SUCCESS;
	}

	VERBOSE = vm.count("verbose");

	// hyper-parameters
	unsigned K = vm["K"].as<unsigned>();
	unsigned beam_size = vm["beam_size"].as<unsigned>();
	float alpha = vm["alpha"].as<float>();
	float gamma_1 = vm["gamma_1"].as<float>();
	float gamma_2 = vm["gamma_2"].as<float>();
	MAX_EPOCH = vm["epoch"].as<unsigned>();
	DEV_ROUND = vm["dev_round"].as<unsigned>();	

	//--- load models
	// Transformer Model recipes
	std::vector<std::shared_ptr<transformer::TransformerModel>> v_tf_models;
	transformer::SentinelMarkers sm;// sentinel markers

	std::vector<std::string> model_paths({vm["model-path-s2t"].as<std::string>(), vm["model-path-t2s"].as<std::string>()});		
	for (auto& model_path : model_paths){
		std::string config_file = model_path + "/model.config";// configuration file path
		struct stat sb;
		if (stat(config_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)){// check existence	
			// load vocabulary from file(s)
			dynet::Dict sd, td;// vocabularies
			std::string vocab_file = model_path + "/" + "src-tgt.joint-vocab";
			if (stat(vocab_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)){
				load_vocab(model_path + "/" + "src-tgt.joint-vocab", sd);
				td = sd;
			}
			else{
				std::string src_vocab_file = model_path + "/" + "src.vocab";
				std::string tgt_vocab_file = model_path + "/" + "tgt.vocab";
				load_vocabs(src_vocab_file, tgt_vocab_file, sd, td);
			}

			sd.freeze();
			td.freeze();
			
			sm._kSRC_SOS = sd.convert("<s>");
			sm._kSRC_EOS = sd.convert("</s>");
			sm._kTGT_SOS = td.convert("<s>");
			sm._kTGT_EOS = td.convert("</s>");

			// read model configuration
			transformer::TransformerConfig tfc;
			ifstream inpf_cfg(config_file);
			assert(inpf_cfg);
		
			std::string line;
			getline(inpf_cfg, line);
			std::stringstream ss(line);
			tfc._src_vocab_size = sd.size();
			tfc._tgt_vocab_size = td.size();
			tfc._sm = sm;
			ss >> tfc._num_units >> tfc._nheads >> tfc._nlayers >> tfc._n_ff_units_factor
			   >> tfc._encoder_emb_dropout_rate >> tfc._encoder_sublayer_dropout_rate >> tfc._decoder_emb_dropout_rate >> tfc._decoder_sublayer_dropout_rate >> tfc._attention_dropout_rate >> tfc._ff_dropout_rate 
			   >> tfc._use_label_smoothing >> tfc._label_smoothing_weight
			   >> tfc._position_encoding >> tfc._position_encoding_flag >> tfc._max_length
			   >> tfc._attention_type
			   >> tfc._ffl_activation_type
			   >> tfc._shared_embeddings
			   >> tfc._use_hybrid_model;
			tfc._is_training = true;
			tfc._use_dropout = true;

			// load models
			v_tf_models.push_back(std::shared_ptr<transformer::TransformerModel>());
			v_tf_models.back().reset(new transformer::TransformerModel(tfc, sd, td));
			std::string model_file = model_path + "/model.params";
			if (stat(model_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode))
			{
				cerr << "Loading pre-trained model from file: " << model_file << "..." << endl;
				v_tf_models.back()->initialise_params_from_file(model_file);// load pre-trained model (for incremental training)
			}
			cerr << "Count of model parameters: " << v_tf_models.back()->get_model_parameters().parameter_count() << endl;
		}
		else TRANSFORMER_RUNTIME_ASSERT("Failed to load model(s) from: " + std::string(model_path) + "!");
	}

	// Transformer LModel recipes
	std::vector<std::shared_ptr<transformer::TransformerLModel>> v_tf_lmodels;
	model_paths.clear();
	model_paths.insert(model_paths.begin(), {vm["model-path-s"].as<std::string>(), vm["model-path-t"].as<std::string>()});
	for (auto& model_path : model_paths){
		std::string config_file = model_path + "/model.config";// configuration file path
		struct stat sb;
		if (stat(config_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)){// check existence	
			// load vocabulary from files
			dynet::Dict d;
			std::string vocab_file = model_path + "/" + "vocab";
			load_vocab(vocab_file, d);

			// load config file
			transformer::TransformerConfig tfc;
			tfc._tgt_vocab_size = d.size();
			tfc._sm = sm;

			ifstream inpf_cfg(config_file);
			assert(inpf_cfg);
			std::string line;
			getline(inpf_cfg, line);
			std::stringstream ss(line);
			ss >> tfc._num_units >> tfc._nheads >> tfc._nlayers >> tfc._n_ff_units_factor
			   >> tfc._decoder_emb_dropout_rate >> tfc._decoder_sublayer_dropout_rate >> tfc._attention_dropout_rate >> tfc._ff_dropout_rate 
			   >> tfc._use_label_smoothing >> tfc._label_smoothing_weight
			   >> tfc._position_encoding >> tfc._max_length
			   >> tfc._attention_type
			   >> tfc._ffl_activation_type
			   >> tfc._use_hybrid_model;		
			tfc._is_training = false;// language models will be fixed?
			tfc._use_dropout = false;

			// load models
			v_tf_lmodels.push_back(std::shared_ptr<transformer::TransformerLModel>());
			v_tf_lmodels.back().reset(new transformer::TransformerLModel(tfc, d));
			std::string model_file = model_path + "/model.params";
			if (stat(model_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode))
			{
				cerr << "Loading pre-trained model from file: " << model_file << "..." << endl;
				v_tf_lmodels.back()->initialise_params_from_file(model_file);// load pre-trained model (for incremental training)
			}
			cerr << "Count of model parameters: " << v_tf_lmodels.back()->get_model_parameters().parameter_count() << endl;
		}
	}
	
	//--- load data
	// monolingual corpora
	// Assume that these monolingual corpora use the same vocabularies with parallel corpus	used for training.
	MonoData mono_cor_s, mono_cor_t;
	cerr << endl << "Reading monolingual source data from " << vm["mono_s"].as<string>() << "...\n";
	mono_cor_s = read_mono_data(v_tf_lmodels[0]->get_dict(), vm["mono_s"].as<string>(), vm["max-seq-len"].as<unsigned>(), vm["mono-load-percent"].as<unsigned>());
	cerr << "Reading monolingual target data from " << vm["mono_t"].as<string>() << "...\n";
	mono_cor_t = read_mono_data(v_tf_lmodels[1]->get_dict(), vm["mono_t"].as<string>(), vm["max-seq-len"].as<unsigned>(), vm["mono-load-percent"].as<unsigned>());

	// development parallel data
	WordIdCorpus devel_cor;// integer-converted dev parallel data
	cerr << endl << "Reading dev parallel data from " << vm["devel"].as<std::string>() << "...\n";
	devel_cor = read_corpus(vm["devel"].as<std::string>(), &v_tf_models[0]->get_source_dict(), &v_tf_models[0]->get_target_dict(), false/*for development*/);

	//--- execute round tripping
	run_round_tripping(v_tf_models, v_tf_lmodels, 
				mono_cor_s, mono_cor_t,
				devel_cor,
				K, beam_size, alpha, gamma_1, gamma_2,
				0/*use normal SGD*/);
	
	
	// finished!
	return EXIT_SUCCESS;
}

void run_round_tripping(std::vector<std::shared_ptr<transformer::TransformerModel>>& v_tm_models, std::vector<std::shared_ptr<transformer::TransformerLModel>>& v_alm_models
                , const MonoData& mono_s, const MonoData& mono_t
                , const WordIdCorpus& dev_cor // for evaluation
                , unsigned K, unsigned beam_size, float alpha, float gamma_1, float gamma_2 /*hyper-parameters of round tripping framework*/
                , unsigned opt_type)
{
	cerr << "Performing round tripping learning..." << endl;

	// get dicts
	dynet::Dict& sd = v_tm_models[0]->get_source_dict();
	dynet::Dict& td = v_tm_models[0]->get_target_dict();

	// set up monolingual data
	vector<unsigned> orders_s(mono_s.size());// IDs from mono_s
	vector<unsigned> orders_t(mono_t.size());// IDs from mono_t
	shuffle(orders_s.begin(), orders_s.end(), *rndeng);// to make it random
	shuffle(orders_t.begin(), orders_t.end(), *rndeng);

	// set up optimizers 
	dynet::Trainer* p_sgd_s2t = create_sgd_trainer(opt_type, gamma_1, v_tm_models[0]->get_model_parameters());
	dynet::Trainer* p_sgd_t2s = create_sgd_trainer(opt_type, gamma_2, v_tm_models[1]->get_model_parameters());

	// pointers for switching between the models
	transformer::TransformerModel *p_tf_s2t = nullptr, *p_tf_t2s = nullptr;
	transformer::TransformerLModel *p_alm = nullptr;

	// start the round tripping algorithm	
	unsigned long id_s = 0, id_t = 0;
	unsigned r = 0/*round*/, epoch_s2t = 0, epoch_t2s = 0;
	bool flag = true;// role of source and target
	unsigned cpt_s2t = 0, cpt_t2s = 0/*count of patience*/;
	while (epoch_s2t < MAX_EPOCH 
		|| epoch_t2s < MAX_EPOCH)// FIXME: simple stopping criterion, another?
	{
		dynet::ComputationGraph cg; 

		if (id_s == orders_s.size()){
			shuffle(orders_s.begin(), orders_s.end(), *rndeng);// to make it random
			id_s = 0;// reset id
			epoch_s2t++;// FIXME: adjust the learning rate if required?
		}
		if (id_t == orders_t.size()){
			shuffle(orders_t.begin(), orders_t.end(), *rndeng);// to make it random
			id_t = 0;// reset id
			epoch_t2s++;// FIXME: adjust the learning rate if required?
		}

		// sample sentence sentA and sentB from mono_cor_s and mono_cor_s respectively
		WordIdSentence sent;
		if (flag){// sample from A
			sent = mono_s[orders_s[id_s++]];
			p_tf_s2t = v_tm_models[0].get();
			p_tf_t2s = v_tm_models[1].get();
			p_alm = v_alm_models[1].get();
		}
		else{// sample from B
			sent = mono_t[orders_t[id_t++]];
			p_tf_s2t = v_tm_models[1].get();
			p_tf_t2s = v_tm_models[0].get();
			p_alm = v_alm_models[0].get();
		}

		//---
		cerr << "Sampled sentence: " << get_sentence(sent, (flag?sd:td)) << endl;
		//---

		// generate K translated sentences s_{mid,1},...,s_{mid,K} using beam search according to translation model P(.|sentA; mod_am_s2t).
		std::vector<WordIdSentence> v_mid_hyps;
		cerr << "Performing beam decoding..." << endl;
		p_tf_s2t->beam_decode(cg, sent, v_mid_hyps, beam_size, K);
		std::vector<dynet::Expression> v_r1, v_r2;
		for (auto& mid_hyp : v_mid_hyps){
			cerr << "Decoded sentence: " << get_sentence(mid_hyp, (flag?td:sd)) << endl;		

			// set the language-model reward for current sampled sentence from p_alm
			auto r1 = p_alm->build_graph(cg, WordIdSentences(1, mid_hyp));
			
			// set the communication reward for current sampled sentence from p_tf_t2s
			auto r2 = p_tf_t2s->build_graph(cg, WordIdSentences(1, mid_hyp), WordIdSentences(1, sent));
			
			// interpolate the rewards
			auto r = alpha * r1 + (1.f - alpha) * r2;
			v_r1.push_back(r * (p_tf_s2t->build_graph(cg, WordIdSentences(1, sent), WordIdSentences(1, mid_hyp))));
			v_r2.push_back((1.f - alpha) * r2);
		}

		// set total loss function
		dynet::Expression i_loss_s2t = dynet::sum(v_r1) / K;// use dynet::average(v_r1) instead?
		dynet::Expression i_loss_t2s = dynet::sum(v_r2) / K;// use dynet::average(v_r2) instead?
		dynet::Expression i_loss = i_loss_s2t + i_loss_t2s;

		// execute forward step
		cg.incremental_forward(i_loss);		
		//-----------------------------------
		float loss = dynet::as_scalar(cg.get_value(i_loss)), loss_s2t = dynet::as_scalar(cg.get_value(i_loss_s2t)), loss_t2s = dynet::as_scalar(cg.get_value(i_loss_t2s));
		p_sgd_s2t->status(); p_sgd_t2s->status();
		cerr << "round=" << r << "; " << "id_s=" << id_s << "; " << "id_t=" << id_t << "; " << "loss=" << loss << "; " << "loss_s2t=" << loss_s2t << "; " << "loss_t2s=" << loss_t2s << endl;
		//-----------------------------------
		
		// execute backward step (including computation of derivatives)
		cg.backward(i_loss);

		// update parameters
		p_sgd_s2t->update();
		p_sgd_t2s->update();		

		// switch source and target roles
		flag = !flag;

		if (id_s == id_t) r++;

		// testing over the development data to check the improvements (after a desired number of rounds)
		if (r == DEV_ROUND){
			// clear the graph first
			cg.clear();

			/*
			ModelStats dstats_s2t, dstats_t2s;
			for (unsigned i = 0; i < dev_cor.size(); ++i) {
				WordIdSentence ssent, tsent;
				tie(ssent, tsent) = dev_cor[i]; 
			
				auto i_xent_s2t = p_tf_s2t->build_graph(cg, WordIdSentences(1, ssent), WordIdSentences(1, tsent), &dstats_s2t);
				auto i_xent_t2s = p_tf_t2s->build_graph(cg, WordIdSentences(1, tsent), WordIdSentences(1, ssent), &dstats_t2s);

				dstats_s2t._scores[1] += dynet::as_scalar(cg.incremental_forward(i_xent_s2t));
				dstats_t2s._scores[1] += dynet::as_scalar(cg.incremental_forward(i_xent_t2s));
			}

			dstats_s2t.update_best_score(cpt_s2t);
			dstats_t2s.update_best_score(cpt_t2s);

			if (cpt_s2t == 0) p_tf_s2t->save_params_to_file(params_out_file);
			if (cpt_t2s == 0) p_tf_t2s->save_params_to_file(params_out_file);			

			cerr << "--------------------------------------------------------------------------------------------------------" << endl;
			cerr << "***DEV (s2t) [epoch=" << epoch_s2t + (float)id_s/(float)orders_s.size() << " eta=" << p_sgd_s2t->learning_rate << "]" << " sents=" << dev_cor.size() << " src_unks=" << dstats_s2t._words_src_unk << " trg_unks=" << dstats_s2t._words_tgt_unk <<  << endl;
			cerr << "***DEV (t2s) [epoch=" << epoch_t2s + (float)id_t/(float)orders_t.size() << " eta=" << p_sgd_t2s->learning_rate << "]" << " sents=" << dev_cor.size() << " src_unks=" << dstats_t2s.words_src_unk << " trg_unks=" << dstats_t2s.words_tgt_unk << " E=" << (dstats_t2s.loss / dstats_t2s.words_tgt) << " ppl=" << exp(dstats_t2s.loss / dstats_t2s.words_tgt) << endl;
cerr << "--------------------------------------------------------------------------------------------------------" << endl;
			*/

			r = 0;
		}
	}
}

// ---
dynet::Trainer* create_sgd_trainer(unsigned opt_type, float lr, dynet::ParameterCollection& model)
{
	// setup SGD trainer
	Trainer* sgd = nullptr;
	unsigned sgd_type = opt_type;
	if (sgd_type == 1)
		sgd = new dynet::MomentumSGDTrainer(model, lr);
	else if (sgd_type == 2)
		sgd = new dynet::AdagradTrainer(model, lr);
	else if (sgd_type == 3)
		sgd = new dynet::AdadeltaTrainer(model);
	else if (sgd_type == 4)
		sgd = new dynet::AdamTrainer(model, lr);
	else if (sgd_type == 5)
		sgd = new dynet::RMSPropTrainer(model, lr);
	else if (sgd_type == 0)//Vanilla SGD trainer
		sgd = new dynet::SimpleSGDTrainer(model, lr);
	else
	   	TRANSFORMER_RUNTIME_ASSERT("Unknown SGD trainer type! (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp)");
	return sgd;
}
// ---

MonoData read_mono_data(dynet::Dict& d, const string &filepath, unsigned max_seq_len, unsigned load_percent)
{
	MonoData mono = read_corpus(filepath, &d, true, max_seq_len);
	
	// limit the percent of training data to be used
	if (load_percent < 100 
		&& load_percent > 0)
	{
		cerr << "Only use " << load_percent << "% of " << mono.size() << " instances: ";
		unsigned int rev_pos = load_percent * mono.size() / 100;
		mono.erase(mono.begin() + rev_pos, mono.end());
		cerr << mono.size() << " instances remaining!" << endl;
	}
	else if (load_percent != 100){
		cerr << "Invalid --mono-load-percent <num> used. <num> must be (0,100]" << endl;
		cerr << "All data will be used!" << endl;
		return mono;// return full
	}

	return mono;
}

//---
std::string get_sentence(const WordIdSentence& source, Dict& td){
	WordId eos_sym = td.convert("</s>");

	std::stringstream ss;
	for (WordId w : source){
		if (w == eos_sym) {
			ss << "</s>";
			break;// stop if seeing EOS marker!
		}

		ss << td.convert(w) << " ";
	}

	return ss.str();
}
//---


