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

unsigned MINIBATCH_SIZE = 512;

unsigned SAMPLE_SIZE = 2;

bool VERBOSE;

int main_body(variables_map vm);

typedef WordIdSentences MonoData;

void get_dev_stats(const WordIdCorpus &devel_cor
	, const transformer::TransformerConfig& tfc
	, transformer::ModelStats& dstats
	, bool swap=false);
std::string get_sentence(const WordIdSentence& source, Dict& td);

// eval on dev (batch supported)
void eval_on_dev_batch(transformer::TransformerModel &tf, 
	const std::vector<WordIdSentences> &dev_src_minibatch, const std::vector<WordIdSentences> &dev_tgt_minibatch,  
	transformer::ModelStats& dstats, 
	unsigned dev_eval_mea, unsigned dev_eval_infer_algo);

// create SGD trainer
dynet::Trainer* create_sgd_trainer(unsigned opt_type, float lr, dynet::ParameterCollection& model);

// read the data
MonoData read_mono_data(dynet::Dict& d, const string &filepath, unsigned max_seq_len=0, unsigned load_percent=100);

// main round tripping function
void run_round_tripping(std::vector<transformer::TransformerModel*>& v_tm_models, std::vector<transformer::TransformerLModel*>& v_alm_models
	, dynet::Trainer*& p_sgd_s2t, dynet::Trainer*& p_sgd_t2s
	, const MonoData& mono_s, const MonoData& mono_t
	, const WordIdCorpus& train_cor, const WordIdCorpus& dev_cor // for evaluation
	, unsigned K, unsigned beam_size, float alpha, float gamma_1, float gamma_2 /*hyper-parameters of round tripping framework*/
	, unsigned dev_eval_mea);

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
		("train,t", value<string>(), "file containing real parallel sentences, with "
			"each line consisting of source ||| target.")
		("max-seq-len", value<unsigned>()->default_value(0), "limit the sequence length loaded from mono data; none by default")
		("mono-load-percent", value<unsigned>()->default_value(100), "limit the use of <num> percent of sequences in monolingual data; full by default")
		//-----------------------------------------	
		("K", value<unsigned>()->default_value(2), "the K value for sampling K-best translations from transformer models")
		("beam_size", value<unsigned>()->default_value(4), "the beam size of beam search decoding")
		("alpha", value<float>()->default_value(0.01f), "the alpha hyper-parameter for balancing the rewards")
		("gamma_1", value<float>()->default_value(0.001f), "the gamma 1 hyper-parameter for stochastic gradient update in tuning source-to-target transformer model")
		("gamma_2", value<float>()->default_value(0.001f), "the gamma 2 hyper-parameter for stochastic gradient update in tuning target-to-source transformer model")
		//-----------------------------------------
		("dev-eval-measure", value<unsigned>()->default_value(0), "specify measure for evaluating dev data during training (0: perplexity; 1: BLEU; 2: NIST; 3: WER; 4: RIBES); default 0 (perplexity)")
		//-----------------------------------------
		("mono_s", value<string>()->default_value(""), "File to read the monolingual source from")
		("mono_t", value<string>()->default_value(""), "File to read the monolingual target from")
		("sample_size", value<unsigned>()->default_value(SAMPLE_SIZE), "sampling size from monolingual source and target data")
		//-----------------------------------------
		("minibatch_size,b", value<unsigned>()->default_value(MINIBATCH_SIZE), "minibatch size for training and development data")
		("epoch,e", value<unsigned>()->default_value(20), "number of training epochs, 50 by default")
		("dev_round", value<unsigned>()->default_value(10000), "number of rounds for evaluating over development data, 25000 by default")
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
	MINIBATCH_SIZE = vm["minibatch_size"].as<unsigned>();
	SAMPLE_SIZE = vm["sample_size"].as<unsigned>();

	//--- load models
	// Transformer Model recipes
	std::vector<transformer::TransformerModel*> v_tf_models;
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
			tfc._model_path = model_path;

			// load models
			v_tf_models.push_back(new transformer::TransformerModel(tfc, sd, td));
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
	std::vector<transformer::TransformerLModel*> v_tf_lmodels;
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
			v_tf_lmodels.push_back(new transformer::TransformerLModel(tfc, d));
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

	// real train parallel data
	WordIdCorpus train_cor;// integer-converted training parallel data
	cerr << endl << "Reading real training parallel data from " << vm["train"].as<std::string>() << "...\n";
	train_cor = read_corpus(vm["train"].as<std::string>(), &v_tf_models[0]->get_source_dict(), &v_tf_models[0]->get_target_dict(), true/*for training*/);

	// development parallel data
	WordIdCorpus devel_cor;// integer-converted dev parallel data
	cerr << endl << "Reading development parallel data from " << vm["devel"].as<std::string>() << "...\n";
	devel_cor = read_corpus(vm["devel"].as<std::string>(), &v_tf_models[0]->get_source_dict(), &v_tf_models[0]->get_target_dict(), false/*for development*/);

	// set up optimizers 
	unsigned opt_type = 0;// normal SGD
	dynet::ParameterCollection& mod_s2t = (*v_tf_models[0]).get_model_parameters();
	dynet::ParameterCollection& mod_t2s = (*v_tf_models[1]).get_model_parameters();
	dynet::Trainer* p_sgd_s2t = create_sgd_trainer(opt_type, gamma_1, mod_s2t);
	dynet::Trainer* p_sgd_t2s = create_sgd_trainer(opt_type, gamma_2, mod_t2s);

	//--- execute round tripping
	run_round_tripping(v_tf_models, v_tf_lmodels, 
				p_sgd_s2t, p_sgd_t2s,
				mono_cor_s, mono_cor_t,
				train_cor, devel_cor,
				K, beam_size, alpha, gamma_1, gamma_2,
				vm["dev-eval-measure"].as<unsigned>());

	// release memory
	delete p_sgd_s2t;
	delete p_sgd_t2s;
	for (auto& mod : v_tf_models) delete mod;// FIXME: use std::shapred_ptr instead of raw pointers
	for (auto& mod : v_tf_lmodels) delete mod;
	v_tf_models.clear();
	v_tf_lmodels.clear();
		
	// finished!
	return EXIT_SUCCESS;
}

void run_round_tripping(std::vector<transformer::TransformerModel*>& v_tm_models, std::vector<transformer::TransformerLModel*>& v_alm_models
	, dynet::Trainer*& p_sgd_s2t, dynet::Trainer*& p_sgd_t2s
	, const MonoData& mono_s, const MonoData& mono_t
        , const WordIdCorpus& train_cor, const WordIdCorpus& dev_cor // for evaluation
        , unsigned K, unsigned beam_size, float alpha, float gamma_1, float gamma_2 /*hyper-parameters of round tripping framework*/
	, unsigned dev_eval_mea)
{
	cerr << endl << "Performing round tripping learning..." << endl;

	// get dicts
	dynet::Dict& sd = v_tm_models[0]->get_source_dict();
	dynet::Dict& td = v_tm_models[0]->get_target_dict();

	// configs
	const transformer::TransformerConfig& tfc_s2t = v_tm_models[0]->get_config();
	const transformer::TransformerConfig& tfc_t2s = v_tm_models[1]->get_config();

	// set up monolingual data
	vector<unsigned> orders_s(mono_s.size());// IDs from mono_s
	vector<unsigned> orders_t(mono_t.size());// IDs from mono_t
	std::iota(orders_s.begin(), orders_s.end(), 0);
	std::iota(orders_t.begin(), orders_t.end(), 0);
	shuffle(orders_s.begin(), orders_s.end(), *rndeng);// to make it random
	shuffle(orders_t.begin(), orders_t.end(), *rndeng);

	// model stats on dev
	transformer::ModelStats dstats_s2t(dev_eval_mea);
	transformer::ModelStats dstats_t2s(dev_eval_mea);
	get_dev_stats(dev_cor, tfc_s2t, dstats_s2t);
	get_dev_stats(dev_cor, tfc_t2s, dstats_t2s, true);

	// create minibatches for training and development data
	// train
	std::vector<std::vector<WordIdSentence> > train_src_minibatch, train_trg_minibatch;
	create_minibatches(train_cor, 1024/*set it by default*/, train_src_minibatch, train_trg_minibatch);// on train
	std::vector<size_t> train_ids_minibatch;
	// create a sentence list for this train minibatch
	train_ids_minibatch.resize(train_src_minibatch.size());
	std::iota(train_ids_minibatch.begin(), train_ids_minibatch.end(), 0);
	// shuffle minibatches
	std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);
	unsigned tid = 0;

	// dev
	std::vector<std::vector<WordIdSentence> > dev_src_minibatch, dev_trg_minibatch;
	create_minibatches(dev_cor, 1024/*set it by default*/, dev_src_minibatch, dev_trg_minibatch);// on dev

	unsigned cpt_s2t = 0, cpt_t2s = 0/*count of patience*/;
	
	eval_on_dev_batch(*v_tm_models[0], dev_src_minibatch, dev_trg_minibatch, dstats_s2t, 0, 0);// batched version (2-3 times faster)
	eval_on_dev_batch(*v_tm_models[1], dev_trg_minibatch, dev_src_minibatch, dstats_t2s, 0, 0);// batched version (2-3 times faster)
	
	dstats_s2t.update_best_score(cpt_s2t);
	dstats_t2s.update_best_score(cpt_t2s);
	
	cerr << "--------------------------------------------------------------------------------------------------------" << endl;
	cerr << "Pre-trained model scores on dev data..." << endl;
	cerr << "***DEV (s2t): " << "sents=" << dev_cor.size() << " src_unks=" << dstats_s2t._words_src_unk << " trg_unks=" << dstats_s2t._words_tgt_unk << " " << dstats_s2t.get_score_string(false) << endl;
	cerr << "***DEV (t2s): " << "sents=" << dev_cor.size() << " src_unks=" << dstats_t2s._words_src_unk << " trg_unks=" << dstats_t2s._words_tgt_unk << " " << dstats_t2s.get_score_string(false) << endl;
        cerr << "--------------------------------------------------------------------------------------------------------" << endl;

	// start the round tripping algorithm	
	bool sample_flag = true;
	unsigned sample_size = SAMPLE_SIZE;
	unsigned long id_s = 0, id_t = 0;
	unsigned r = 1/*round*/, epoch_s2t = 0, epoch_t2s = 0, total_round = 0;
	bool flag = true;// role of source and target
	while (epoch_s2t < MAX_EPOCH 
		|| epoch_t2s < MAX_EPOCH)// FIXME: simple stopping criterion, another?
	{
		{// this block to prevent multiple graph creation which DyNet does not support yet!
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

			if (tid >= train_ids_minibatch.size()){
				tid = 0;
				std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);
			}

			transformer::TransformerModel*& p_tf_s2t = flag?v_tm_models[0]:v_tm_models[1];
			transformer::TransformerModel*& p_tf_t2s = flag?v_tm_models[1]:v_tm_models[0];
			transformer::TransformerLModel*& p_alm = flag?v_alm_models[1]:v_alm_models[0];

			// sample sentences from real parallel data
			WordIdSentences r_src_sents, r_trg_sents;
			if (sample_flag){// TODO: this is to simulate the warm-up strategy. This flag may be disabled after several iterations. 
				r_src_sents = flag?train_src_minibatch[train_ids_minibatch[tid]]:train_trg_minibatch[train_ids_minibatch[tid]];
				r_trg_sents = flag?train_trg_minibatch[train_ids_minibatch[tid]]:train_src_minibatch[train_ids_minibatch[tid]];
				tid++;
			}

			// sample sentences from monolingual source and target data respectively
			if (r_src_sents.size() > 0) sample_size = r_src_sents.size();
			for (unsigned sid = 0; sid < sample_size; sid++){
		       		auto& sent = flag?mono_s[orders_s[id_s++]]:mono_t[orders_t[id_t++]];
		
				//---
				if (VERBOSE)
					cerr << "Sampled sentence from monolingual data: " << get_sentence(sent, (flag?sd:td)) << endl;
				//---
				
				// generate K translated sentences using beam search according to source-to-target translation model.
				if (VERBOSE) cerr << "Performing beam decoding..." << endl;
				p_tf_s2t->set_dropout(false);// disable dropout for performing beam search
				std::vector<WordIdSentence> trg_sents;
				p_tf_s2t->beam_decode(cg, sent, trg_sents, beam_size, K);
				p_tf_s2t->set_dropout(true);// enable dropout for training

				// mix-up with real training data
				for (auto& tsent : trg_sents){
					//---
					if (VERBOSE)
						cerr << "Decoded sentence: " << get_sentence(tsent, (flag?td:sd)) << endl;
					//---

					if (tsent.size() > 2) { // good hypothesis
						r_src_sents.push_back(sent);
						r_trg_sents.push_back(tsent);
					}
				}
			}
					
			// set the language-model reward for current sampled sentences from p_alm
			auto reward_lm = p_alm->build_graph(cg, r_trg_sents);
			
			// set the communication reward for current sampled sentences from p_tf_t2s
			auto reward_rev = p_tf_t2s->build_graph(cg, r_trg_sents, r_src_sents);
			
			// interpolate the rewards
			auto reward = alpha * reward_lm + (1.f - alpha) * reward_rev;
			auto i_loss_s2t = reward * (p_tf_s2t->build_graph(cg, r_src_sents, r_trg_sents));// need averaging?
			auto i_loss_t2s = (1.f - alpha) * reward_rev;// need averaging?

			// set total loss function
			dynet::Expression i_loss = i_loss_s2t + i_loss_t2s;// final loss

			// execute forward step
			cg.incremental_forward(i_loss);		
			//-----------------------------------
			float loss = dynet::as_scalar(cg.get_value(i_loss.i)), loss_s2t = dynet::as_scalar(cg.get_value(i_loss_s2t.i)), loss_t2s = dynet::as_scalar(cg.get_value(i_loss_t2s.i));
			p_sgd_s2t->status(); //p_sgd_t2s->status();
			cerr << "round=" << total_round << "; " << "id_s=" << id_s << "; " << "id_t=" << id_t << "; " << "loss=" << loss << "; " << "loss_s2t=" << loss_s2t << "; " << "loss_t2s=" << loss_t2s << endl;
			cerr << "-----------------------------------" << endl;
			//-----------------------------------
		
			// execute backward step (including computation of derivatives)
			cg.backward(i_loss);

			// update parameters
			p_sgd_s2t->update();
			p_sgd_t2s->update();	
		}	

		// switch source and target roles
		flag = !flag;

		if (id_s == id_t){
			r++;
			total_round++;
		}

		// evaluate over the development data to check the improvements (after a desired number of rounds)
		if (r % DEV_ROUND == 0){			
			eval_on_dev_batch(*v_tm_models[0], dev_src_minibatch, dev_trg_minibatch, dstats_s2t, 0, 0);// batched version (2-3 times faster)
			eval_on_dev_batch(*v_tm_models[1], dev_trg_minibatch, dev_src_minibatch, dstats_t2s, 0, 0);// batched version (2-3 times faster)

			dstats_s2t.update_best_score(cpt_s2t);
			dstats_t2s.update_best_score(cpt_t2s);

			if (cpt_s2t == 0) v_tm_models[0]->save_params_to_file(tfc_s2t._model_path + "/model.params.rt");
			if (cpt_t2s == 0) v_tm_models[1]->save_params_to_file(tfc_t2s._model_path + "/model.params.rt");

			cerr << "--------------------------------------------------------------------------------------------------------" << endl;
			cerr << "***DEV (s2t) [epoch=" << epoch_s2t + (float)id_s/(float)orders_s.size() << " eta=" << p_sgd_s2t->learning_rate << "]" << " sents=" << dev_cor.size() << " src_unks=" << dstats_s2t._words_src_unk << " trg_unks=" << dstats_s2t._words_tgt_unk << " " << dstats_s2t.get_score_string() << endl;
			cerr << "***DEV (t2s) [epoch=" << epoch_t2s + (float)id_t/(float)orders_t.size() << " eta=" << p_sgd_t2s->learning_rate << "]" << " sents=" << dev_cor.size() << " src_unks=" << dstats_t2s._words_src_unk << " trg_unks=" << dstats_t2s._words_tgt_unk << " " << dstats_t2s.get_score_string() << endl;
			cerr << "--------------------------------------------------------------------------------------------------------" << endl;

			// FIXME: observe cpt_s2t and cpt_t2s
			// ...

			r = 1;
		}
	}
}

void eval_on_dev_batch(transformer::TransformerModel &tf, 
	const std::vector<WordIdSentences> &dev_src_minibatch, const std::vector<WordIdSentences> &dev_tgt_minibatch,  
	transformer::ModelStats& dstats, 
	unsigned dev_eval_mea, unsigned dev_eval_infer_algo)
{
	tf.set_dropout(false);// disable dropout for evaluaing on dev

	if (dev_eval_mea == 0) // perplexity
	{
		double losses = 0.f;
		for (unsigned i = 0; i < dev_src_minibatch.size(); i++) {		
			const auto& ssentb = dev_src_minibatch[i];
			const auto& tsentb = dev_tgt_minibatch[i];

			dynet::ComputationGraph cg;
			auto i_xent = tf.build_graph(cg, ssentb, tsentb, nullptr, true);
			losses += as_scalar(cg.forward(i_xent));
		}

		dstats._scores[1] = losses;
	}
	else{
		// FIXME
	}
	
	tf.set_dropout(true);
}
// ---

// ---
void get_dev_stats(const WordIdCorpus &devel_cor
	, const transformer::TransformerConfig& tfc
	, transformer::ModelStats& dstats
	, bool swap)
{
	for (unsigned i = 0; i < devel_cor.size(); ++i) {
		WordIdSentence ssent, tsent;
		tie(ssent, tsent) = devel_cor[i];  

		if (swap) std::swap(ssent, tsent);

		dstats._words_src += ssent.size();
		dstats._words_tgt += tsent.size() - 1; // shifted right 
		for (auto& word : ssent) if (word == tfc._sm._kSRC_UNK) dstats._words_src_unk++;
		for (auto& word : tsent) if (word == tfc._sm._kTGT_UNK) dstats._words_tgt_unk++;
	}
}
// ---

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
	if (load_percent < 100 && load_percent > 0)
	{
		cerr << "Only use " << load_percent << "% of " << mono.size() << " instances: ";
	        unsigned int rev_pos = load_percent * mono.size() / 100;
	        mono.erase(mono.begin() + rev_pos, mono.end());
	        cerr << mono.size() << " instances remaining!" << endl;
	}
	else if (load_percent != 100){
		cerr << "Invalid --mono-load-percent <num> used. <num> must be (0,100]" << endl;
	        cerr << "All data will be used!" << endl;
	}

	return mono;// return full                                                                                                                                                                         }
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


