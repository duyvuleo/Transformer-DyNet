/* This is an implementation of Transformer architecture from https://arxiv.org/abs/1706.03762 (Attention is All You need).
* Developed by Cong Duy Vu Hoang
* Updated: 1 Nov 2017
*/

#include "transformer.h"

// STL
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <sys/stat.h>

// Boost
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

// MTEval
#include <mteval/utils.h>
#include <mteval/Evaluator.h>
#include <mteval/EvaluatorFactory.h>
#include <mteval/Statistics.h>

using namespace std;
using namespace dynet;
using namespace transformer;
using namespace MTEval;
using namespace boost::program_options;

// hyper-paramaters for training
unsigned MINIBATCH_SIZE = 1;

bool DEBUGGING_FLAG = false;

bool PRINT_GRAPHVIZ = false;

unsigned TREPORT = 50;
unsigned DREPORT = 5000;

bool SAMPLING_TRAINING = false;

bool VERBOSE = false;

// ---
bool load_data(const variables_map& vm
	, WordIdCorpus& train_cor, WordIdCorpus& devel_cor
	, dynet::Dict& sd, dynet::Dict& td
	, SentinelMarkers& sm);
// ---

// ---
void save_config(const std::string& config_out_file
	, const std::string& params_out_file
	, const TransformerConfig& tfc);
// ---

//---
std::string get_sentence(const WordIdSentence& source, Dict& td);
//---

// ---
dynet::Trainer* create_sgd_trainer(const variables_map& vm, dynet::ParameterCollection& model);
// ---

// ---
void run_train(transformer::TransformerModel &tf, const WordIdCorpus &train_cor, const WordIdCorpus &devel_cor, 
	Trainer &sgd, 
	const std::string& model_path, 
	unsigned max_epochs, unsigned patience, 
	unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience,
	unsigned average_checkpoints, 
	unsigned dev_eval_mea, unsigned dev_eval_infer_algo);// support batching
// ---

// ---
void get_dev_stats(const WordIdCorpus &devel_cor
	, const transformer::TransformerConfig& tfc
	, transformer::ModelStats& dstats);
void eval_on_dev(transformer::TransformerModel &tf, 
	const WordIdCorpus &devel_cor, 
	transformer::ModelStats& dstats, 
	unsigned dev_eval_mea, unsigned dev_eval_infer_algo);
// ---

//************************************************************************************************************************************************************
int main(int argc, char** argv) {
	cerr << "*** DyNet initialization ***" << endl;
	auto dyparams = dynet::extract_dynet_params(argc, argv);
	dynet::initialize(dyparams);	

	// command line processing
	variables_map vm; 
	options_description opts("Allowed options");
	opts.add_options()
		("help", "print help message")
		("config,c", value<std::string>(), "config file specifying additional command line options")
		//-----------------------------------------
		("train,t", value<std::vector<std::string>>(), "file containing training sentences, with each line consisting of source ||| target.")		
		("devel,d", value<std::string>(), "file containing development sentences.")
		("max-seq-len", value<unsigned>()->default_value(0), "limit the sentence length (either source or target); none by default")
		("src-vocab", value<std::string>()->default_value(""), "file containing source vocabulary file; none by default (will be built from train file)")
		("tgt-vocab", value<std::string>()->default_value(""), "file containing target vocabulary file; none by default (will be built from train file)")
		("joint-vocab", value<std::string>()->default_value(""), "file containing target joint vocabulary file for both source and target; none by default (will be built from train file)")
		("train-percent", value<unsigned>()->default_value(100), "use <num> percent of sentences in training data; full by default")
		//-----------------------------------------
		("shared-embeddings", "use shared source and target embeddings (in case that source and target use the same vocabulary; none by default")
		//-----------------------------------------
		("minibatch-size,b", value<unsigned>()->default_value(1), "impose the minibatch size for training (support both GPU and CPU); single batch by default")
		("dynet-autobatch", value<unsigned>()->default_value(0), "impose the auto-batch mode (support both GPU and CPU); no by default")
		//-----------------------------------------
		("sgd-trainer", value<unsigned>()->default_value(0), "use specific SGD trainer (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp; 6: cyclical SGD)")
		("sparse-updates", value<bool>()->default_value(true), "enable/disable sparse update(s) for lookup parameter(s); true by default")
		("grad-clip-threshold", value<float>()->default_value(5.f), "use specific gradient clipping threshold (https://arxiv.org/pdf/1211.5063.pdf); 5 by default")
		//-----------------------------------------
		("model-path,p", value<std::string>()->default_value("."), "all files related to the model will be saved in this folder")
		//-----------------------------------------
		("nlayers", value<unsigned>()->default_value(6), "use <num> layers for stacked encoder/decoder layers; 6 by default")
		("num-units,u", value<unsigned>()->default_value(512), "use <num> dimensions for number of units; 512 by default")
		("num-heads,h", value<unsigned>()->default_value(8), "use <num> for number of heads in multi-head attention mechanism; 4 by default")
		("n-ff-units-factor", value<unsigned>()->default_value(4), "use <num> times of input dim for output dim in feed-forward layer; 4 by default")
		//-----------------------------------------
		("encoder-emb-dropout-p", value<float>()->default_value(0.1f), "use dropout for encoder embeddings; 0.1 by default")
		("encoder-sublayer-dropout-p", value<float>()->default_value(0.1f), "use dropout for sub-layer's output in encoder; 0.1 by default")
		("decoder-emb-dropout-p", value<float>()->default_value(0.1f), "use dropout for decoding embeddings; 0.1 by default")
		("decoder-sublayer-dropout-p", value<float>()->default_value(0.1f), "use dropout for sub-layer's output in decoder; 0.1 by default")
		("attention-dropout-p", value<float>()->default_value(0.1f), "use dropout for attention; 0.1 by default")
		("ff-dropout-p", value<float>()->default_value(0.1f), "use dropout for feed-forward layer; 0.1 by default")
		//-----------------------------------------
		("use-label-smoothing", "use label smoothing for cross entropy; no by default")
		("label-smoothing-weight", value<float>()->default_value(0.1f), "impose label smoothing weight in objective function; 0.1 by default")
		//-----------------------------------------
		("ff-activation-type", value<unsigned>()->default_value(1), "impose feed-forward activation type (1: RELU, 2: SWISH, 3: SWISH with learnable beta); 1 by default")
		//-----------------------------------------
		("position-encoding", value<unsigned>()->default_value(2), "impose positional encoding (0: none; 1: learned positional embedding; 2: sinusoid encoding); 2 by default")
		("position-encoding-flag", value<unsigned>()->default_value(0), "which both (0) / encoder only (1) / decoder only (2) will be applied positional encoding; both (0) by default")
		("max-pos-seq-len", value<unsigned>()->default_value(300), "specify the maximum word-based sentence length (either source or target) for learned positional encoding; 300 by default")
		//-----------------------------------------
		("use-hybrid-model", "use hybrid model in which RNN encodings of source and target are used in place of word embeddings and positional encodings (a hybrid architecture between AM and Transformer?) partially adopted from GNMT style; no by default")
		//-----------------------------------------
		("attention-type", value<unsigned>()->default_value(1), "impose attention type (1: Luong attention type; 2: Bahdanau attention type); 1 by default")
		//-----------------------------------------
		("epochs,e", value<unsigned>()->default_value(20), "maximum number of training epochs")
		("patience", value<unsigned>()->default_value(0), "no. of times in which the model has not been improved for early stopping; default none")
		//-----------------------------------------
		("lr-eta", value<float>()->default_value(0.1f), "SGD learning rate value (e.g., 0.1 for simple SGD trainer or smaller 0.001 for ADAM trainer)")
		("lr-eta-decay", value<float>()->default_value(2.0f), "SGD learning rate decay value")
		//-----------------------------------------
		("lr-epochs", value<unsigned>()->default_value(0), "no. of epochs for starting learning rate annealing (e.g., halving)") // learning rate scheduler 1
		("lr-patience", value<unsigned>()->default_value(0), "no. of times in which the model has not been improved, e.g., for starting learning rate annealing (e.g., halving)") // learning rate scheduler 2
		//-----------------------------------------
		("sampling", "sample translation during training; default not")
		//-----------------------------------------
		("dev-eval-measure", value<unsigned>()->default_value(0), "specify measure for evaluating dev data during training (0: perplexity; 1: BLEU; 2: NIST; 3: WER; 4: RIBES); default 0 (perplexity)") // note that MT scores here are approximate (e.g., evaluating with <unk> markers, and tokenized text or with subword segmentation if using BPE), not necessarily equivalent to real BLEU/NIST/WER/RIBES scores.
		("dev-eval-infer-algo", value<unsigned>()->default_value(1), "specify the algorithm for inference on dev (0: sampling; 1: greedy; N>=2: beam search with N size of beam); default 0 (sampling)") // using sampling/greedy will be faster. 
		//-----------------------------------------
		("average-checkpoints", value<unsigned>()->default_value(1), "specify number of checkpoints for model averaging; default single best model") // average checkpointing
		//-----------------------------------------
		("r2l-target", "use right-to-left direction for target during training; default not")
		//-----------------------------------------
		("swap", "swap roles of source and target, i.e., learn p(source|target)")
		//-----------------------------------------
		("treport", value<unsigned>()->default_value(50), "no. of training instances for reporting current model status on training data")
		("dreport", value<unsigned>()->default_value(5000), "no. of training instances for reporting current model status on development data (dreport = N * treport)")
		//-----------------------------------------
		("print-graphviz", "print graphviz-style computation graph; default not")
		//-----------------------------------------
		("verbose,v", "be extremely chatty")
		//-----------------------------------------
		("debug", "enable/disable simpler debugging by immediate computing mode or checking validity (refers to http://dynet.readthedocs.io/en/latest/debugging.html)")// for CPU only
		("dynet-profiling", value<int>()->default_value(0), "enable/disable auto profiling (https://github.com/clab/dynet/pull/1088/commits/bc34db98fa5e2e694f54f0e6b1d720d517c7530e)")// for debugging only			
	;
	
	store(parse_command_line(argc, argv, opts), vm); 
	if (vm.count("config") > 0)
	{
		ifstream config(vm["config"].as<std::string>().c_str());
		store(parse_config_file(config, opts), vm); 
	}
	notify(vm);

	// print command line
	cerr << endl << "PID=" << ::getpid() << endl;
	cerr << "Command: ";
	for (int i = 0; i < argc; i++){ 
		cerr << argv[i] << " "; 
	} 
	cerr << endl;
	
	// print help
	if (vm.count("help") 
		|| !(vm.count("train") && vm.count("devel")))
	{
		cout << opts << "\n";
		return EXIT_FAILURE;
	}

	// hyper-parameters for training
	DEBUGGING_FLAG = vm.count("debug");
	VERBOSE = vm.count("verbose");
	TREPORT = vm["treport"].as<unsigned>(); 
	DREPORT = vm["dreport"].as<unsigned>(); 
	SAMPLING_TRAINING = vm.count("sampling");
	PRINT_GRAPHVIZ = vm.count("print-graphviz");
	MINIBATCH_SIZE = vm["minibatch-size"].as<unsigned>();

	// get and check model path
	std::string model_path = vm["model-path"].as<std::string>();
	struct stat sb;
	if (stat(model_path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))
		cerr << endl << "All model files will be saved to: " << model_path << "." << endl;
	else
		TRANSFORMER_RUNTIME_ASSERT("The model-path does not exist!");

	// model recipe
	dynet::Dict sd, td;// vocabularies
	SentinelMarkers sm;// sentinel markers
	WordIdCorpus train_cor, devel_cor;// integer-converted train and dev data
	transformer::TransformerConfig tfc;// Transformer's configuration (either loaded from file or newly-created)

	std::string config_file = model_path + "/model.config";// configuration file path
	if (stat(config_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)){// check existence		
		// (incremental training)
		// to load the training profiles from previous training run
		cerr << "Found existing (trained) model from " << model_path << "!" << endl;

		// load vocabulary from files
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

		// initalise sentinel markers
		sm._kSRC_SOS = sd.convert("<s>");
		sm._kSRC_EOS = sd.convert("</s>");
		sm._kTGT_SOS = td.convert("<s>");
		sm._kTGT_EOS = td.convert("</s>");

		// load data files
		if (!load_data(vm, train_cor, devel_cor, sd, td, sm))
			TRANSFORMER_RUNTIME_ASSERT("Failed to load data files!");

		// read model configuration
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
	}
	else{// not exist, meaning that the model will be created from scratch!
		cerr << "Preparing to train the model from scratch..." << endl;

		// load fixed vocabularies from files if provided, otherwise create them on the fly from the training data.
		bool use_joint_vocab = vm.count("joint-vocab");
		if (use_joint_vocab)
			load_joint_vocab(vm["joint-vocab"].as<std::string>(), sd, td);
		else
			load_vocabs(vm["src-vocab"].as<std::string>(), vm["tgt-vocab"].as<std::string>(), sd, td);

		// initalise sentinel markers
		sm._kSRC_SOS = sd.convert("<s>");
		sm._kSRC_EOS = sd.convert("</s>");
		sm._kTGT_SOS = td.convert("<s>");
		sm._kTGT_EOS = td.convert("</s>");

		// load data files
		if (!load_data(vm, train_cor, devel_cor, sd, td, sm))
			TRANSFORMER_RUNTIME_ASSERT("Failed to load data files!");

		// get transformer configuration
		tfc = transformer::TransformerConfig(sd.size(), td.size()
			, vm["num-units"].as<unsigned>()
			, vm["num-heads"].as<unsigned>()
			, vm["nlayers"].as<unsigned>()
			, vm["n-ff-units-factor"].as<unsigned>()
			, vm["encoder-emb-dropout-p"].as<float>()
			, vm["encoder-sublayer-dropout-p"].as<float>()
			, vm["decoder-emb-dropout-p"].as<float>()
			, vm["decoder-sublayer-dropout-p"].as<float>()
			, vm["attention-dropout-p"].as<float>()
			, vm["ff-dropout-p"].as<float>()
			, vm.count("use-label-smoothing")
			, vm["label-smoothing-weight"].as<float>()
			, vm["position-encoding"].as<unsigned>()
			, vm["position-encoding-flag"].as<unsigned>()
			, vm["max-pos-seq-len"].as<unsigned>()
			, sm
			, vm["attention-type"].as<unsigned>()
			, vm["ff-activation-type"].as<unsigned>()
			, use_joint_vocab | vm.count("shared-embeddings")
			, vm.count("use-hybrid-model"));

		// save vocabularies to files
		if (use_joint_vocab){
			std::string vocab_file = model_path + "/" + "src-tgt.joint-vocab";
			save_vocab(vocab_file, sd);
		}
		else{
			std::string src_vocab_file = model_path + "/" + "src.vocab";
			std::string tgt_vocab_file = model_path + "/" + "tgt.vocab";
			save_vocabs(src_vocab_file, tgt_vocab_file, sd, td);
		}

		// save configuration file (for decoding/inference)
		std::string config_out_file = model_path + "/model.config";
		std::string params_out_file = model_path + "/model.params";
		save_config(config_out_file, params_out_file, tfc);
	}	

	// learning rate scheduler
	unsigned lr_epochs = vm["lr-epochs"].as<unsigned>(), lr_patience = vm["lr-patience"].as<unsigned>();
	if (lr_epochs > 0 && lr_patience > 0)
		cerr << "[WARNING] - Conflict on learning rate scheduler; use either lr-epochs or lr-patience!" << endl;

	// initialise transformer object
	transformer::TransformerModel tf(tfc, sd, td);
	std::string model_file = model_path + "/model.params";
	if (stat(model_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode))
	{
		cerr << endl << "Loading pre-trained model from file: " << model_file << "..." << endl;
		tf.initialise_params_from_file(model_file);// load pre-trained model (for incremental training)
	}
	cerr << endl << "Count of model parameters: " << tf.get_model_parameters().parameter_count() << endl;

	// create SGD trainer
	Trainer* p_sgd_trainer = create_sgd_trainer(vm, tf.get_model_parameters());

	if (vm["dev-eval-measure"].as<unsigned>() > 4) TRANSFORMER_RUNTIME_ASSERT("Unknown dev-eval-measure type (0: perplexity; 1: BLEU; 2: NIST; 3: WER; 4: RIBES)!");

	// train transformer model
	run_train(tf
		, train_cor, devel_cor
		, *p_sgd_trainer
		, model_path
		, vm["epochs"].as<unsigned>(), vm["patience"].as<unsigned>() /*early stopping*/
		, lr_epochs, vm["lr-eta-decay"].as<float>(), lr_patience/*learning rate scheduler*/
		, vm["average-checkpoints"].as<unsigned>()
		, vm["dev-eval-measure"].as<unsigned>(), vm["dev-eval-infer-algo"].as<unsigned>());

	// clean up
	cerr << "Cleaning up..." << endl;
	delete p_sgd_trainer;
	// transformer object will be automatically cleaned, no action required!

	return EXIT_SUCCESS;
}
//************************************************************************************************************************************************************

// ---
bool load_data(const variables_map& vm
	, WordIdCorpus& train_cor, WordIdCorpus& devel_cor
	, dynet::Dict& sd, dynet::Dict& td
	, SentinelMarkers& sm)
{
	bool swap = vm.count("swap");
	bool r2l_target = vm.count("r2l_target");

	std::vector<std::string> train_paths = vm["train"].as<std::vector<std::string>>();// to handle multiple training data
	if (train_paths.size() > 2) TRANSFORMER_RUNTIME_ASSERT("Invalid -t or --train parameter. Only maximum 2 training corpora provided!");	
	cerr << endl << "Reading training data from " << train_paths[0] << "...\n";
	if (vm.count("shared-embeddings"))
		train_cor = read_corpus(train_paths[0], &sd, &sd, true, vm["max-seq-len"].as<unsigned>(), r2l_target & !swap);
	else
		train_cor = read_corpus(train_paths[0], &sd, &td, true, vm["max-seq-len"].as<unsigned>(), r2l_target & !swap);
	if ("" == vm["src-vocab"].as<std::string>() 
		&& "" == vm["tgt-vocab"].as<std::string>()) // if not using external vocabularies
	{
		sd.freeze(); // no new word types allowed
		td.freeze(); // no new word types allowed
	}
	if (train_paths.size() == 2)// incremental training
	{
		train_cor.clear();// use the next training corpus instead!	
		cerr << "Reading extra training data from " << train_paths[1] << "...\n";
		train_cor = read_corpus(train_paths[1], &sd, &td, true/*for training*/, vm["max-seq-len"].as<unsigned>(), r2l_target & !swap);
		cerr << "Performing incremental training..." << endl;
	}

	// limit the percent of training data to be used
	unsigned train_percent = vm["train-percent"].as<unsigned>();
	if (train_percent < 100 
		&& train_percent > 0)
	{
		cerr << "Only use " << train_percent << "% of " << train_cor.size() << " training instances: ";
		unsigned int rev_pos = train_percent * train_cor.size() / 100;
		train_cor.erase(train_cor.begin() + rev_pos, train_cor.end());
		cerr << train_cor.size() << " instances remaining!" << endl;
	}
	else if (train_percent != 100){
		cerr << "Invalid --train-percent <num> used. <num> must be (0,100]" << endl;
		return false;
	}

	if (DREPORT >= train_cor.size())
		cerr << "WARNING: --dreport <num> (" << DREPORT << ")" << " is too large, <= training data size (" << train_cor.size() << ")" << endl;

	// set up <unk> ids
	sd.set_unk("<unk>");
	sm._kSRC_UNK = sd.get_unk_id();
	td.set_unk("<unk>");
	sm._kTGT_UNK = td.get_unk_id();

	if (vm.count("devel")) {
		cerr << "Reading dev data from " << vm["devel"].as<std::string>() << "...\n";
		devel_cor = read_corpus(vm["devel"].as<std::string>(), &sd, &td, false/*for development*/, 0, r2l_target & !swap);
	}

	if (swap) {
		cerr << "Swapping role of source and target\n";
		if (!vm.count("shared-embeddings")){
			std::swap(sd, td);
			std::swap(sm._kSRC_SOS, sm._kTGT_SOS);
			std::swap(sm._kSRC_EOS, sm._kTGT_EOS);
			std::swap(sm._kSRC_UNK, sm._kTGT_UNK);
		}

		for (auto &sent: train_cor){
			std::swap(get<0>(sent), get<1>(sent));
			if (r2l_target){
				WordIdSentence &tsent = get<1>(sent);
				std::reverse(tsent.begin() + 1, tsent.end() - 1);
			}
		}
		
		for (auto &sent: devel_cor){
			std::swap(get<0>(sent), get<1>(sent));
			if (r2l_target){
				WordIdSentence &tsent = get<1>(sent);
				std::reverse(tsent.begin() + 1, tsent.end() - 1);
			}
		}
	}

	return true;
}
// ---

// ---
dynet::Trainer* create_sgd_trainer(const variables_map& vm, dynet::ParameterCollection& model){
	// setup SGD trainer
	Trainer* sgd = nullptr;
	unsigned sgd_type = vm["sgd-trainer"].as<unsigned>();
	if (sgd_type == 1)
		sgd = new MomentumSGDTrainer(model, vm["lr-eta"].as<float>());
	else if (sgd_type == 2)
		sgd = new AdagradTrainer(model, vm["lr-eta"].as<float>());
	else if (sgd_type == 3)
		sgd = new AdadeltaTrainer(model);
	else if (sgd_type == 4)
		sgd = new AdamTrainer(model, vm["lr-eta"].as<float>());
	else if (sgd_type == 5)
		sgd = new RMSPropTrainer(model, vm["lr-eta"].as<float>());
	else if (sgd_type == 0)//Vanilla SGD trainer
		sgd = new SimpleSGDTrainer(model, vm["lr-eta"].as<float>());
	else
	   	TRANSFORMER_RUNTIME_ASSERT("Unknown SGD trainer type! (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp)");
	sgd->clip_threshold = vm["grad-clip-threshold"].as<float>();// * MINIBATCH_SIZE;// use larger gradient clipping threshold if training with mini-batching, correct?
	sgd->sparse_updates_enabled = vm["sparse-updates"].as<bool>();
	if (!sgd->sparse_updates_enabled)
		cerr << "Sparse updates for lookup parameter(s) to be disabled!" << endl;

	return sgd;
}
// ---

// ---
void get_dev_stats(const WordIdCorpus &devel_cor
	, const transformer::TransformerConfig& tfc
	, transformer::ModelStats& dstats) // ToDo: support batch?
{
	for (unsigned i = 0; i < devel_cor.size(); ++i) {
		WordIdSentence ssent, tsent;
		tie(ssent, tsent) = devel_cor[i];  

		dstats._words_src += ssent.size();
		dstats._words_tgt += tsent.size() - 1; // shifted right 
		for (auto& word : ssent) if (word == tfc._sm._kSRC_UNK) dstats._words_src_unk++;
		for (auto& word : tsent) if (word == tfc._sm._kTGT_UNK) dstats._words_tgt_unk++;
	}
}

void eval_on_dev(transformer::TransformerModel &tf, 
	const WordIdCorpus &devel_cor, 
	transformer::ModelStats& dstats, 
	unsigned dev_eval_mea, unsigned dev_eval_infer_algo)
{
	if (dev_eval_mea == 0) // perplexity
	{
		double losses = 0.f;
		for (unsigned i = 0; i < devel_cor.size(); ++i) {
			WordIdSentence ssent, tsent;
			tie(ssent, tsent) = devel_cor[i];  

			dynet::ComputationGraph cg;
			auto i_xent = tf.build_graph(cg, WordIdSentences(1, ssent), WordIdSentences(1, tsent), nullptr, true);
			losses += as_scalar(cg.forward(i_xent));
		}

		dstats._scores[1] = losses;
	}
	else{
		// create evaluators
		std::string spec;
		if (dev_eval_mea == 1) spec = "BLEU";
		else if (dev_eval_mea == 2) spec = "NIST";
		else if (dev_eval_mea == 3) spec = "WER";
		else if (dev_eval_mea == 4) spec = "RIBES";
		std::shared_ptr<MTEval::Evaluator> evaluator(MTEval::EvaluatorFactory::create(spec));
		std::vector<MTEval::Sample> v_samples;
		for (unsigned i = 0; i < devel_cor.size(); ++i) {
			WordIdSentence ssent, tsent;
			tie(ssent, tsent) = devel_cor[i];  

			// inference
			dynet::ComputationGraph cg;
			WordIdSentence thyp;// raw translation (w/o scores)
			if (dev_eval_infer_algo == 0)// random sampling
				tf.sample(cg, ssent, thyp);// fastest with bad translations
			else if (dev_eval_infer_algo == 1)// greedy decoding
				tf.greedy_decode(cg, ssent, thyp);// faster with relatively good translations
			else// beam search decoding
				tf.beam_decode(cg, ssent, thyp, dev_eval_infer_algo/*N>1: beam decoding with N size of beam*/);// slow with better translations
					
			// collect statistics for mteval
			v_samples.push_back(MTEval::Sample({thyp, {tsent/*, tsent2, tsent3, tsent4*/}}));// multiple references are supported as well!
      			evaluator->prepare(v_samples[v_samples.size() - 1]);
		}
		
		// analyze the evaluation score
		MTEval::Statistics eval_stats;
    		for (unsigned i = 0; i < v_samples.size(); ++i) {			
      			eval_stats += evaluator->map(v_samples[i]);
    		}

		dstats._scores[1] = evaluator->integrate(eval_stats);
	}
}
// ---

// ---
void run_train(transformer::TransformerModel &tf, const WordIdCorpus &train_cor, const WordIdCorpus &devel_cor, 
	Trainer &sgd, 
	const std::string& model_path,
	unsigned max_epochs, unsigned patience, 
	unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience,
	unsigned average_checkpoints,
	unsigned dev_eval_mea, unsigned dev_eval_infer_algo)
{
	// get current configuration
	const transformer::TransformerConfig& tfc = tf.get_config();

	// model params file
	std::string params_out_file = model_path + "/model.params";

	// create minibatches
	std::vector<std::vector<WordIdSentence> > train_src_minibatch;
	std::vector<std::vector<WordIdSentence> > train_trg_minibatch;
	std::vector<size_t> train_ids_minibatch;
	size_t minibatch_size = MINIBATCH_SIZE;
	create_minibatches(train_cor, minibatch_size, train_src_minibatch, train_trg_minibatch, train_ids_minibatch);
  
	// model stats on dev
	transformer::ModelStats dstats(dev_eval_mea);
	get_dev_stats(devel_cor, tfc, dstats);
	
	unsigned report_every_i = TREPORT;
	unsigned dev_every_i_reports = DREPORT;

	// shuffle minibatches
	cerr << endl << "***SHUFFLE" << endl;
	std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

	unsigned sid = 0, id = 0, last_print = 0;
	MyTimer timer_epoch("completed in"), timer_iteration("completed in");
	unsigned epoch = 0, cpt = 0/*count of patience*/;
	while (epoch < max_epochs) {
		transformer::ModelStats tstats;

		tf.set_dropout(true);// enable dropout

		for (unsigned iter = 0; iter < dev_every_i_reports;) {
			if (id == train_ids_minibatch.size()) { 
				//timing
				cerr << "***Epoch " << epoch << " is finished. ";
				timer_epoch.show();

				epoch++;

				id = 0;
				sid = 0;
				last_print = 0;

				// learning rate scheduler 1: after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.
				if (lr_epochs > 0 && epoch >= lr_epochs)
					sgd.learning_rate /= lr_eta_decay; 

				if (epoch >= max_epochs) break;

				// shuffle the access order
				cerr << "***SHUFFLE" << endl;
				std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);				

				timer_epoch.reset();
			}

			// build graph for this instance
			dynet::ComputationGraph cg;// dynamic computation graph for each data batch
			if (DEBUGGING_FLAG){//http://dynet.readthedocs.io/en/latest/debugging.html
				cg.set_immediate_compute(true);
				cg.set_check_validity(true);
			}
	
			transformer::ModelStats ctstats;
			Expression i_xent = tf.build_graph(cg, train_src_minibatch[train_ids_minibatch[id]], train_trg_minibatch[train_ids_minibatch[id]], &ctstats);
	
			if (PRINT_GRAPHVIZ) {
				cerr << "***********************************************************************************" << endl;
				cg.print_graphviz();
				cerr << "***********************************************************************************" << endl;
			}

			Expression i_objective = i_xent;

			// perform forward computation for aggregate objective
			cg.forward(i_objective);

			// grab the parts of the objective
			float loss = as_scalar(cg.get_value(i_xent.i));
			if (!is_valid(loss)){
				std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				++id;
				continue;
			}

			tstats._scores[1] += loss;
			tstats._words_src += ctstats._words_src;
			tstats._words_src_unk += ctstats._words_src_unk;  
			tstats._words_tgt += ctstats._words_tgt;
			tstats._words_tgt_unk += ctstats._words_tgt_unk;  

			cg.backward(i_objective);
			sgd.update();

			sid += train_trg_minibatch[train_ids_minibatch[id]].size();
			iter += train_trg_minibatch[train_ids_minibatch[id]].size();

			if (sid / report_every_i != last_print 
					|| iter >= dev_every_i_reports
					|| id + 1 == train_ids_minibatch.size()){
				last_print = sid / report_every_i;

				float elapsed = timer_iteration.elapsed();

				sgd.status();
				cerr << "sents=" << sid << " ";
				cerr /*<< "loss=" << tstats._scores[1]*/ << "src_unks=" << tstats._words_src_unk << " trg_unks=" << tstats._words_tgt_unk << " " << tstats.get_score_string() << ' ';// << " E=" << (tstats._scores[1] / tstats._words_tgt) << " ppl=" << exp(tstats._scores[1] / tstats._words_tgt) << ' ';
				cerr /*<< "time_elapsed=" << elapsed*/ << "(" << (float)(tstats._words_src + tstats._words_tgt) * 1000.f / elapsed << " words/sec)" << endl; 	
			}
			   		 
			++id;
		}

		timer_iteration.reset();

		// show score on dev data?
		tf.set_dropout(false);// disable dropout for evaluating dev data

		// sample a random sentence (for observing translations during training progress)
		if (SAMPLING_TRAINING){// Note: this will slow down the training process, suitable for debugging only.
			dynet::ComputationGraph cg;
			WordIdSentence target;// raw translation (w/o scores)
			cerr << endl << "---------------------------------------------------------------------------------------------------" << endl;
			cerr << "***Source: " << get_sentence(train_src_minibatch[train_ids_minibatch[id]][0], tf.get_source_dict()) << endl;
			tf.sample(cg, train_src_minibatch[train_ids_minibatch[id]][0], target);
			cerr << "***Sampled translation: " << get_sentence(target, tf.get_target_dict()) << endl;
			cg.clear();
			tf.greedy_decode(cg, train_src_minibatch[train_ids_minibatch[id]][0], target);
			cerr << "***Greedy translation: " << get_sentence(target, tf.get_target_dict()) << endl;
			cerr << "---------------------------------------------------------------------------------------------------" << endl << endl;
		}
		
		eval_on_dev(tf, devel_cor, dstats, dev_eval_mea, dev_eval_infer_algo);
		dstats.update_best_score(cpt);
		if (cpt == 0){
			// FIXME: consider average checkpointing?
			tf.save_params_to_file(params_out_file);
		}

		cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		cerr << "***DEV [epoch=" << (float)epoch + (float)sid/(float)train_cor.size() << " eta=" << sgd.learning_rate << "]" << " sents=" << devel_cor.size() << " src_unks=" << dstats._words_src_unk << " trg_unks=" << dstats._words_tgt_unk << " " << dstats.get_score_string() << ' ';

		if (cpt > 0) cerr << "(not improved, best score on dev so far: " << dstats.get_score_string(false) << ") ";
		timer_iteration.show();

		// learning rate scheduler 2: if the model has not been improved for lr_patience times, decrease the learning rate by lr_eta_decay factor.
		if (lr_patience > 0 && cpt > 0 && cpt % lr_patience == 0){
			cerr << "The model has not been improved for " << lr_patience << " times. Decreasing the learning rate..." << endl;
			sgd.learning_rate /= lr_eta_decay;
		}

		// another early stopping criterion
		if (patience > 0 && cpt >= patience)
		{
			cerr << "The model has not been improved for " << patience << " times. Stopping now...!" << endl;
			cerr << "No. of epochs so far: " << epoch << "." << endl;
			cerr << "Best score on dev: " << dstats.get_score_string(false) << endl;
			cerr << "--------------------------------------------------------------------------------------------------------" << endl;
			break;
		}
		cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		timer_iteration.reset();
	}

	cerr << endl << "Transformer training completed!" << endl;
}
// ---

//---
std::string get_sentence(const WordIdSentence& source, Dict& td){
	std::stringstream ss;
	for (WordId w : source){
		ss << td.convert(w) << " ";
	}

	return ss.str();
}
//---

//---
void save_config(const std::string& config_out_file, const std::string& params_out_file, const TransformerConfig& tfc)
{
	// each line has the format: 
	// <num-units> <num-heads> <nlayers> <ff-num-units-factor> <encoder-emb-dropout> <encoder-sub-layer-dropout> <decoder-emb-dropout> <decoder-sublayer-dropout> <attention-dropout> <ff-dropout> <use-label-smoothing> <label-smoothing-weight> <position-encoding-type> <max-seq-len> <attention-type> <ff-activation-type> <use-hybrid-model> <your-trained-model-path>
	// e.g.,
	// 128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 0 300 1 1 0 0 <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010101_att1_ls01_pe1_ml300_ffrelu_run1
	// 128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 0 300 1 1 0 0 <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010101_att1_ls01_pe1_ml300_ffrelu_run2
	std::stringstream ss;
		
	ss << tfc._num_units << " " << tfc._nheads << " " << tfc._nlayers << " " << tfc._n_ff_units_factor << " "
		<< tfc._encoder_emb_dropout_rate << " " << tfc._encoder_sublayer_dropout_rate << " " << tfc._decoder_emb_dropout_rate << " " << tfc._decoder_sublayer_dropout_rate << " " << tfc._attention_dropout_rate << " " << tfc._ff_dropout_rate << " "
		<< tfc._use_label_smoothing << " " << tfc._label_smoothing_weight << " "
		<< tfc._position_encoding << " " << tfc._position_encoding_flag << " " << tfc._max_length << " "
		<< tfc._attention_type << " "
		<< tfc._ffl_activation_type << " "
		<< tfc._shared_embeddings << " "
		<< tfc._use_hybrid_model << " ";		
	ss << params_out_file;

	ofstream outf_cfg(config_out_file);
	assert(outf_cfg);
	outf_cfg << ss.str();
}
//---
