// Transformer enhanced with moment matching training technique

#include "../transformer.h"
#include "mm-features.h"

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

bool RESET_IF_STUCK = false;
bool SWITCH_TO_ADAM = false;
bool USE_SMALLER_MINIBATCH = false;
unsigned NUM_RESETS = 1;

unsigned NUM_SAMPLES = 2;
unsigned SAMPLING_SIZE = 1;

bool VERBOSE = false;
bool DEBUG_FLAG = false;

// ---
bool load_data(const variables_map& vm
	, std::pair<WordIdSentences, WordIdSentences>& mono_cor, WordIdCorpus& devel_cor
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
void sanity_check(transformer::TransformerModel &tf, 
	MMFeatures_UDA& mm_feas/*feature config for moment matching*/, 
	const std::vector<WordIdSentences>& in_s_mono_data_minibatch,
	const std::vector<size_t>& in_s_mono_data_ids_minibatch);
// ---

// ---
void run_train(transformer::TransformerModel &tf, 
	const std::pair<WordIdSentences, WordIdSentences>& in_mono_data, const WordIdCorpus& devel_cor, 
	dynet::Trainer*& p_sgd, 
	unsigned training_mode,
	float alpha,
	float softmax_temp,
	const std::string& model_path, 
	unsigned max_epochs, unsigned patience, 
	unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience,
	unsigned average_checkpoints, 
	unsigned dev_eval_mea, unsigned dev_eval_infer_algo,
	MMFeatures_UDA& mm_feas/*features for moment matching*/);// support batching
// ---

// ---
dynet::Expression compute_mm_score(dynet::ComputationGraph& cg, 
	const std::vector<float>& v_pre_emp_scores,
	const WordIdSentences& ssents, 
	const WordIdSentences& samples,
	unsigned bsize, 
	MMFeatures_UDA& mm_feas);
// ---

// --
dynet::Expression compute_mm_loss(dynet::ComputationGraph& cg,
        const WordIdSentences& ssents,
        transformer::TransformerModel& tf,
        const std::vector<float>& v_pre_mm_scores,
        MMFeatures_UDA& mm_feas,
	float softmax_temp,
	transformer::ModelStats& ctstats,
	dynet::Expression* p_i_mm);
// ---

// ---
void eval_mm_on_dev(transformer::TransformerModel &tf, 
	MMFeatures_UDA& mm_feas,
	const WordIdCorpus &devel_cor, 
	float softmax_temp,
	const std::vector<std::vector<float>>& v_pre_emp_scores,
	transformer::ModelStats& dstats);
// ---

// ---
void get_dev_stats(const WordIdCorpus &devel_cor
	, const transformer::TransformerConfig& tfc
	, transformer::ModelStats& dstats);
void eval_on_dev(transformer::TransformerModel &tf, 
	const WordIdCorpus &devel_cor, 
	transformer::ModelStats& dstats, 
	unsigned dev_eval_mea, unsigned dev_eval_infer_algo);
void eval_on_dev(transformer::TransformerModel &tf, 
	const std::vector<WordIdSentences> &dev_src_minibatch, const std::vector<WordIdSentences> &dev_tgt_minibatch,  
	transformer::ModelStats& dstats, 
	unsigned dev_eval_mea, unsigned dev_eval_infer_algo);// batched version (much faster)
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
		("devel,d", value<std::string>(), "file containing small bilingual in-domain sentences for development purpose")
		("max-seq-len", value<unsigned>()->default_value(0), "limit the sentence length (either source or target); none by default")
		("src-vocab", value<std::string>()->default_value(""), "file containing source vocabulary file; none by default (will be built from train file)")
		("tgt-vocab", value<std::string>()->default_value(""), "file containing target vocabulary file; none by default (will be built from train file)")
		("joint-vocab", value<std::string>(), "file containing target joint vocabulary file for both source and target; none by default (will be built from train file)")
		("train-percent", value<unsigned>()->default_value(100), "use <num> percent of sentences in training data; full by default")
		//-----------------------------------------
		("in-src-mono-data", value<std::string>()->default_value(""), "file containing in-domain source monolingual sentences.")
		("in-tgt-mono-data", value<std::string>()->default_value(""), "file containing in-domain target monolingual sentences.")
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
		("src-alm-model-path", value<std::string>()->default_value(""), "specify model path for transformer source language model")
		("tgt-alm-model-path", value<std::string>()->default_value(""), "specify model path for transformer target language model")
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
		("training-mode", value<unsigned>()->default_value(2), "specify training mode (0: MLE; 1: MM; 2: interleave; 3: mixed); default 2")
		("alpha", value<float>()->default_value(0.017f), "specify alpha value in mixed training mode; default 0.017")
		//-----------------------------------------
		("num-samples", value<unsigned>()->default_value(NUM_SAMPLES), "use <num> of samples produced by the current model; 2 by default")
		("sampling-size", value<unsigned>()->default_value(SAMPLING_SIZE), "sampling size; default 10")
		("softmax-temperature", value<float>()->default_value(1.f), "use temperature for softmax activation; 1.0 by default")
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
		// learning rate scheduler
		("lr-epochs", value<unsigned>()->default_value(0), "no. of epochs for starting learning rate annealing (e.g., halving)") // learning rate scheduler 1
		("lr-patience", value<unsigned>()->default_value(0), "no. of times in which the model has not been improved, e.g., for starting learning rate annealing (e.g., halving)") // learning rate scheduler 2
		//-----------------------------------------
		("reset-if-stuck", "a strategy if the model gets stuck then reset everything and resume training; default not")
		("switch-to-adam", "switch to Adam trainer if getting stuck; default not")
		("use-smaller-minibatch", "use smaller mini-batch size if getting stuck; default not")
		("num-resets", value<unsigned>()->default_value(1), "no. of times the training process will be reset; default 1") 
		//-----------------------------------------
		("sampling", "sample translation during training; default not")
		("mm-debug", "very chatty for debugging only; default not")
		//-----------------------------------------
		("dev-eval-measure", value<unsigned>()->default_value(0), "specify measure for evaluating dev data during training (0: perplexity; 1: BLEU; 2: NIST; 3: WER; 4: RIBES); default 0 (perplexity)") // note that MT scores here are approximate (e.g., evaluating with <unk> markers, and tokenized text or with subword segmentation if using BPE), not necessarily equivalent to real BLEU/NIST/WER/RIBES scores.
		("dev-eval-infer-algo", value<unsigned>()->default_value(0), "specify the algorithm for inference on dev (0: sampling; 1: greedy; N>=2: beam search with N size of beam); default 0 (random sampling)") // using sampling/greedy will be faster. 
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
		|| !(vm.count("in-src-mono-data") && vm.count("in-tgt-mono-data") && vm.count("devel")))
	{
		cout << opts << "\n";
		return EXIT_FAILURE;
	}

	// hyper-parameters for training
	DEBUGGING_FLAG = vm.count("debug");
	VERBOSE = vm.count("verbose");
	DEBUG_FLAG = vm.count("mm-debug");
	TREPORT = vm["treport"].as<unsigned>(); 
	DREPORT = vm["dreport"].as<unsigned>(); 
	SAMPLING_TRAINING = vm.count("sampling");
	PRINT_GRAPHVIZ = vm.count("print-graphviz");
	RESET_IF_STUCK = vm.count("reset-if-stuck");
	SWITCH_TO_ADAM = vm.count("switch-to-adam");
	USE_SMALLER_MINIBATCH = vm.count("use-smaller-minibatch");
	NUM_RESETS = vm["num-resets"].as<unsigned>();
	MINIBATCH_SIZE = vm["minibatch-size"].as<unsigned>();
	NUM_SAMPLES = vm["num-samples"].as<unsigned>();
	SAMPLING_SIZE = vm["sampling-size"].as<unsigned>();

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
	WordIdCorpus devel_cor;// integer-converted bilingual data for development
	pair<WordIdSentences, WordIdSentences> mono_cor;// integer-converted monolingual in-domain data
	transformer::TransformerConfig tfc;// Transformer's configuration (either loaded from file or newly-created)

	std::string config_file = model_path + "/model.config";// configuration file path
	std::string model_file = model_path + "/model.params";
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
		sm._kSRC_UNK = sd.convert("<unk>");
		sm._kTGT_SOS = td.convert("<s>");
		sm._kTGT_EOS = td.convert("</s>");
		sm._kTGT_UNK = td.convert("<unk>");

		// load data files
		if (!load_data(vm, mono_cor, devel_cor, sd, td, sm))
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
		ss >> model_file;
	}
	else{// not exist, meaning that the model will be created from scratch!
		cerr << "Preparing to train the model from scratch..." << endl;

		// load fixed vocabularies from files if provided, otherwise create them on the fly from the training data.
		bool use_joint_vocab = vm.count("joint-vocab") | vm.count("shared-embeddings");
		if (use_joint_vocab)
			load_joint_vocab(vm.count("joint-vocab")?vm["joint-vocab"].as<std::string>():"", sd, td);
		else
			load_vocabs(vm["src-vocab"].as<std::string>(), vm["tgt-vocab"].as<std::string>(), sd, td);

		// initalise sentinel markers
		sm._kSRC_SOS = sd.convert("<s>");
		sm._kSRC_EOS = sd.convert("</s>");
		sm._kSRC_UNK = sd.convert("<unk>");
		sm._kTGT_SOS = td.convert("<s>");
		sm._kTGT_EOS = td.convert("</s>");
		sm._kTGT_UNK = td.convert("<unk>");

		// load data files
		if (!load_data(vm, mono_cor, devel_cor, sd, td, sm))
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
			, use_joint_vocab
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
	//std::string model_file = model_path + "/model.params";
	if (stat(model_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode))
	{
		cerr << endl << "Loading pre-trained model from file: " << model_file << "..." << endl;
		tf.initialise_params_from_file(model_file);// load pre-trained model (for incremental training)
	}
	cerr << "Count of model parameters: " << tf.get_model_parameters().parameter_count() << endl << endl;

	// create SGD trainer
	Trainer* p_sgd_trainer = create_sgd_trainer(vm, tf.get_model_parameters());

	if (vm["dev-eval-measure"].as<unsigned>() > 4) TRANSFORMER_RUNTIME_ASSERT("Unknown dev-eval-measure type (0: perplexity; 1: BLEU; 2: NIST; 3: WER; 4: RIBES)!");

	// external module for moment matching feature computation
	MMFeatures_UDA* p_mm_fea_cfg = new MMFeatures_UDA(NUM_SAMPLES
					, sd, td
					, vm["src-alm-model-path"].as<std::string>(), vm["tgt-alm-model-path"].as<std::string>());/*feature config for moment matching*/

	unsigned training_mode = vm["training-mode"].as<unsigned>();
	float alpha = vm["alpha"].as<float>();
	float softmax_temp = vm["softmax-temperature"].as<float>();

	// train transformer model
	run_train(tf
		, mono_cor, devel_cor
		, p_sgd_trainer
		, training_mode
		, alpha
		, softmax_temp
		, model_path
		, vm["epochs"].as<unsigned>(), vm["patience"].as<unsigned>() /*early stopping*/
		, lr_epochs, vm["lr-eta-decay"].as<float>(), lr_patience/*learning rate scheduler*/
		, vm["average-checkpoints"].as<unsigned>()
		, vm["dev-eval-measure"].as<unsigned>(), vm["dev-eval-infer-algo"].as<unsigned>()
		, *p_mm_fea_cfg);

	// clean up
	cerr << "Cleaning up..." << endl;
	delete p_sgd_trainer;
	delete p_mm_fea_cfg;
	// transformer object will be automatically cleaned, no action required!

	return EXIT_SUCCESS;
}
//************************************************************************************************************************************************************

// ---
bool load_data(const variables_map& vm
	, std::pair<WordIdSentences, WordIdSentences>& mono_cor, WordIdCorpus& devel_cor
	, dynet::Dict& sd, dynet::Dict& td
	, SentinelMarkers& sm)
{
	bool swap = vm.count("swap");
	bool r2l_target = vm.count("r2l_target");

	// source
	std::string in_src_mono_data = vm["in-src-mono-data"].as<std::string>();	
	cerr << endl << "Reading in-domain source monolingual data from " << in_src_mono_data << "...\n";	 
	mono_cor.first = read_corpus(in_src_mono_data, &sd, true, vm["max-seq-len"].as<unsigned>(), r2l_target);
	
	// target
	std::string in_tgt_mono_data = vm["in-tgt-mono-data"].as<std::string>();	
	cerr  << "Reading in-domain target monolingual data from " << in_tgt_mono_data << "...\n";
	mono_cor.second = read_corpus(in_tgt_mono_data, &td, true, vm["max-seq-len"].as<unsigned>(), r2l_target);

	// freeze dicts if required
	if ("" == vm["src-vocab"].as<std::string>() 
		&& "" == vm["tgt-vocab"].as<std::string>()) // if not using external vocabularies
	{
		sd.freeze(); // no new word types allowed
		td.freeze(); // no new word types allowed
	}

	// set up <unk> ids
	sd.set_unk("<unk>");
	sm._kSRC_UNK = sd.get_unk_id();
	td.set_unk("<unk>");
	sm._kTGT_UNK = td.get_unk_id();

	// load parallel dev data
	if (vm.count("devel")) {
		cerr << "Reading dev data from " << vm["devel"].as<std::string>() << "...\n";
		devel_cor = read_corpus(vm["devel"].as<std::string>(), &sd, &td, false/*for development*/, 0, r2l_target & !swap);
	}

	// swap if required
	bool use_joint_vocab = vm.count("joint-vocab") | vm.count("shared-embeddings");
	if (swap) {
		cerr << "Swapping role of source and target\n";
		if (!use_joint_vocab){
			std::swap(sd, td);
			std::swap(sm._kSRC_SOS, sm._kTGT_SOS);
			std::swap(sm._kSRC_EOS, sm._kTGT_EOS);
			std::swap(sm._kSRC_UNK, sm._kTGT_UNK);
		}
		
		for (auto &sent: devel_cor){
			std::swap(get<0>(sent), get<1>(sent));
			if (r2l_target){
				WordIdSentence &tsent = get<1>(sent);
				std::reverse(tsent.begin() + 1, tsent.end() - 1);
			}
		}

		std::swap(mono_cor.first, mono_cor.second);
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
	, transformer::ModelStats& dstats)
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
			const auto& ssent = std::get<0>(devel_cor[i]);
			const auto& tsent = std::get<1>(devel_cor[i]);
			//tie(ssent, tsent) = devel_cor[i];

			// inference
			dynet::ComputationGraph cg;
			WordIdSentence thyp;// raw translation (w/o scores)
			if (dev_eval_infer_algo == 0)// random sampling
				tf.sample(cg, ssent, thyp);// fastest with bad translations
			else if (dev_eval_infer_algo == 1)// greedy decoding
				tf.greedy_decode(cg, ssent, thyp);// faster with relatively good translations
			else// beam search decoding
			{
				WordIdSentences thyps;	
				tf.beam_decode(cg, ssent, thyps, dev_eval_infer_algo/*N>1: beam decoding with N size of beam*/);// slow with better translations
				thyp = thyps[0];
			}
					
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
void eval_on_dev(transformer::TransformerModel &tf, 
	const std::vector<WordIdSentences> &dev_src_minibatch, const std::vector<WordIdSentences> &dev_tgt_minibatch,  
	transformer::ModelStats& dstats, 
	unsigned dev_eval_mea, unsigned dev_eval_infer_algo) // batched version
{
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
		// create evaluators
		std::string spec;
		if (dev_eval_mea == 1) spec = "BLEU";
		else if (dev_eval_mea == 2) spec = "NIST";
		else if (dev_eval_mea == 3) spec = "WER";
		else if (dev_eval_mea == 4) spec = "RIBES";
		std::shared_ptr<MTEval::Evaluator> evaluator(MTEval::EvaluatorFactory::create(spec));
		std::vector<MTEval::Sample> v_samples;
		for (unsigned i = 0; i < dev_src_minibatch.size(); ++i) {
			const auto& ssents = dev_src_minibatch[i];
			const auto& tsents = dev_tgt_minibatch[i];

			// batched inference/decoding
			dynet::ComputationGraph cg;
			WordIdSentences thyps;// raw translation (w/o scores)
			if (dev_eval_infer_algo == 0)// random sampling
				tf.sample(cg, ssents, thyps);// fastest with bad translations
			else if (dev_eval_infer_algo == 1)// greedy decoding
				tf.greedy_decode(cg, ssents, thyps);// faster with relatively good translations
			else// beam search decoding
			{
				// FIXME: current version of beam search does not support batched decoding yet!
				TRANSFORMER_RUNTIME_ASSERT("Current version of beam search does not support batched decoding yet!");
			}
					
			// collect statistics for mteval
			for (unsigned h = 0; h < tsents.size(); h++){
				v_samples.push_back(MTEval::Sample({thyps[h], {tsents[h]/*, tsent2, tsent3, tsent4*/}}));// multiple references are supported as well!
      				evaluator->prepare(v_samples.back());
			}
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
void eval_mm_on_dev(transformer::TransformerModel &tf, 
	MMFeatures_UDA& mm_feas,
        const WordIdCorpus &devel_cor,
	float softmax_temp,
	const std::vector<std::vector<float>>& v_pre_emp_scores,
        transformer::ModelStats& dstats)
{
	unsigned F_dim = v_pre_emp_scores[0].size();

	std::vector<float> v_scores;
	float mm_loss = 0.f;
	unsigned i = 0;
	//cerr << "i=";
	for (; i < devel_cor.size() && i < SAMPLING_SIZE; ++i) {
		/*

		//cerr << i << " ";
		const auto& ssent = std::get<0>(devel_cor[i]);
		//const auto& tsent = std::get<1>(devel_cor[i]);
		
		dynet::ComputationGraph cg;

		// sample from current model
		WordIdSentences samples;
		tf.set_dropout(false);
                tf.sample_sentences(cg, ssent, mm_feas._num_samples, samples, v_scores, softmax_temp);
                tf.set_dropout(true);

		// compute mm loss
		cg.clear();
		v_scores.clear();
		mm_feas.compute_feature_scores(WordIdSentences(mm_feas._num_samples, ssent), samples, v_scores);
		//cerr << "phi_h: ";
		//for (auto& score : v_scores) cerr << score << " ";
		//cerr << endl;

		dynet::Expression i_mod = dynet::input(cg, dynet::Dim({F_dim, 1}, mm_feas._num_samples), v_scores);
		i_mod = dynet::mean_batches(i_mod);

		//cerr << "phi_bar: ";
		//for (auto& score : v_pre_emp_scores[i]) cerr << score << " ";
		//cerr << endl;
		dynet::Expression i_emp = dynet::input(cg, dynet::Dim({F_dim, 1}, 1), v_pre_emp_scores[i]);

		dynet::Expression i_dist = dynet::squared_distance(i_mod, i_emp);
		cg.incremental_forward(i_dist);
		mm_loss += dynet::as_scalar(cg.get_value(i_dist.i));		 
		//cerr << mm_loss << endl;
		
		*/
	}

	//cerr << endl;

        dstats._scores[1] = mm_loss / i;//devel_cor.size();
}
// ---

// ---
dynet::Expression compute_mm_score(dynet::ComputationGraph& cg, 
	const std::vector<float>& v_pre_emp_scores,
	const WordIdSentences& ssents, 
	const WordIdSentences& samples, 
	unsigned bsize,
	MMFeatures_UDA& mm_feas)
{
	std::vector<float> scores;
	mm_feas.compute_feature_scores_on_targets(cg, samples, scores);
	if (DEBUG_FLAG){
		cerr << "scores: ";
		for (auto& score : scores) cerr << score << " ";
		cerr << endl;
	}
	
	// compute moment matching: <\hat{\Phi}(x) - \bar{\Phi}, \Phi(x) - \bar{\Phi}>
	unsigned F_dim = mm_feas._F_dim;
	//cerr << "1" << endl;
	dynet::Expression i_phi_x = dynet::input(cg, dynet::Dim({F_dim, mm_feas._num_samples}, bsize), scores);// ((|F|, |S|), batch_size)
	//cerr << "2" << endl;
	dynet::Expression i_phi_x_csum = dynet::sum_dim(i_phi_x, {1});// ((|F|, 1), batch_size)
	//cerr << "3" << endl;
	i_phi_x_csum = dynet::concatenate_cols(std::vector<dynet::Expression>(mm_feas._num_samples, i_phi_x_csum));// ((|F|, |S|), batch_size)
	//cerr << "4" << endl;
	dynet::Expression i_phi_hat = (i_phi_x_csum - i_phi_x) / (mm_feas._num_samples - 1);// ((|F|, |S|), batch_size)
	//cerr << "5" << endl;
	i_phi_x = dynet::reshape(i_phi_x, dynet::Dim({F_dim, 1}, (unsigned)samples.size()));// ((|F|, 1), batch_size * |S|)
	i_phi_hat = dynet::reshape(i_phi_hat, dynet::Dim({F_dim, 1}, (unsigned)samples.size()));// ((|F|, 1), batch_size * |S|)
	if (DEBUG_FLAG){
		cg.incremental_forward(i_phi_hat);
		cerr << "phi_hat_scores: ";
		std::vector<float> phi_hat_scores = dynet::as_vector(cg.get_value(i_phi_hat.i));
		for (auto& sc : phi_hat_scores)
			cerr << sc << " ";
		cerr << endl;
	}
	dynet::Expression i_phi_bar = dynet::input(cg, dynet::Dim({mm_feas._F_dim, 1}, (unsigned)samples.size()), v_pre_emp_scores);// ((F_dim, 1), batch_size * |S|)
	if (DEBUG_FLAG){
		cg.incremental_forward(i_phi_bar);
		cerr << "phi_bar_scores: ";
		std::vector<float> phi_bar_scores = dynet::as_vector(cg.get_value(i_phi_bar.i));
		for (auto& sc : phi_bar_scores)
			cerr << sc << " ";
		cerr << endl;
	}
	//cerr << "6" << endl;
	//cerr << "i_phi_bar.dim()=" << "((" << i_phi_bar.dim()[0] << "," << i_phi_bar.dim()[1] << "), " << i_phi_bar.dim().batch_elems() << ")" << endl;
	//cerr << "i_phi_x.dim()=" << "((" << i_phi_x.dim()[0] << "," << i_phi_x.dim()[1] << "), " << i_phi_x.dim().batch_elems() << ")" << endl;
	//cerr << "i_phi_hat.dim()=" << "((" << i_phi_hat.dim()[0] << "," << i_phi_hat.dim()[1] << "), " << i_phi_hat.dim().batch_elems() << ")" << endl;
	dynet::Expression i_mm = dynet::dot_product(i_phi_hat - i_phi_bar, i_phi_x - i_phi_bar);// ((1, 1), batch_size * |S|)
	if (DEBUG_FLAG){
		cg.incremental_forward(i_mm);
		cerr << "mm_scores: ";
		std::vector<float> mm_scores = dynet::as_vector(cg.get_value(i_mm.i));
		for (auto& sc : mm_scores)
			cerr << sc << " ";
		cerr << endl;
	}	

	return i_mm;
}
// ---

// ---
dynet::Expression compute_mm_loss(dynet::ComputationGraph& cg, 
	const WordIdSentences& ssents, 
	transformer::TransformerModel& tf,
	const std::vector<float>& v_pre_emp_scores, 
	MMFeatures_UDA& mm_feas,
	float softmax_temp,
	transformer::ModelStats& ctstats,
	dynet::Expression* p_i_mm)
{
	WordIdSentences ssents_ext, samples;
	for (auto& ssent : ssents){
		WordIdSentences results;
		std::vector<float> v_probs;// unused for now
		if (DEBUG_FLAG) cerr << "source: " << get_sentence(ssent, tf.get_source_dict()) << endl;
		tf.set_dropout(false);
		tf.sample_sentences(cg, ssent, NUM_SAMPLES, results, v_probs, softmax_temp);
		tf.set_dropout(true);

		if (DEBUG_FLAG)
			for (auto& sample : results) cerr << "sample: " << get_sentence(sample, tf.get_target_dict()) << endl;

		ssents_ext.insert(ssents_ext.end(), results.size()/*equal to NUM_SAMPLES*/, ssent);
		samples.insert(samples.end(), results.begin(), results.end());
	}
	if (DEBUG_FLAG) cerr << "samples.size()=" << samples.size() << endl;

	// compute moment matching scores		
	//dynet::Expression i_mm = compute_mm_score(cg, v_pre_emp_scores, ssents_ext, samples, ssents.size(), mm_feas);// shape=((1,1), batch_size * |S|)
	*p_i_mm = compute_mm_score(cg, v_pre_emp_scores, ssents_ext, samples, ssents.size(), mm_feas);// shape=((1,1), batch_size * |S|)

	// compute loss associated with mm scores
	//return tf.build_graph(cg, ssents_ext, samples, i_mm, &ctstats);// reinforced CE loss
	//return tf.build_graph(cg, ssents_ext, samples, &ctstats);// conventional CE loss
	return tf.get_all_losses(cg, ssents_ext, samples, &ctstats);// conventional CE loss (all individual losses)
}
// ---

// ---
void sanity_check(transformer::TransformerModel &tf, 
	MMFeatures_UDA& mm_feas/*feature config for moment matching*/, 
	const std::vector<WordIdSentences>& in_s_mono_data_minibatch,
	const std::vector<size_t>& in_s_mono_data_ids_minibatch)
{
	tf.set_dropout(false);// disable dropout

	float score_phi_bar = 0.f;
	unsigned num_samples = 0;
	for (unsigned s = 0; s < in_s_mono_data_minibatch.size() && num_samples < SAMPLING_SIZE; ++s) 
	{
		const auto& ssents = in_s_mono_data_minibatch[in_s_mono_data_ids_minibatch[s]];

		// get greedy translations
		dynet::ComputationGraph cg;
		WordIdSentences thyps;
		tf.greedy_decode(cg, ssents, thyps);// faster with relatively good translations
		
		// compute score
		std::vector<float> tmp_scores;
		mm_feas.compute_feature_scores_on_targets(cg, thyps, tmp_scores, true);
		score_phi_bar += tmp_scores[0];
		
		num_samples += thyps.size();	
	}
	score_phi_bar /= num_samples;// averaging
	if (DEBUG_FLAG) cerr << "score_phi_bar=" << score_phi_bar << endl;

	tf.set_dropout(true);// disable dropout
}
// ---

// ---
void run_train(transformer::TransformerModel &tf, 
	const std::pair<WordIdSentences, WordIdSentences>& in_mono_data, const WordIdCorpus &devel_cor, 
	dynet::Trainer*& p_sgd, 
	unsigned training_mode,
	float alpha,
	float softmax_temp,
	const std::string& model_path,
	unsigned max_epochs, unsigned patience, 
	unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience,
	unsigned average_checkpoints,
	unsigned dev_eval_mea, unsigned dev_eval_infer_algo, 
	MMFeatures_UDA& mm_feas/*feature config for moment matching*/)
{
	// get current configuration
	const transformer::TransformerConfig& tfc = tf.get_config();

	// model params file
	std::stringstream ss;
	if (training_mode == 0)
		ss << model_path << "/model.0.params";
	else
		ss << model_path << "/model.mm." << training_mode << ".a" << alpha << ".smte" << softmax_temp << mm_feas.get_name() << ".params";
	std::string params_out_file = ss.str();
	//std::string params_out_file = model_path + "/model.mm.params";// save to different file with pre-trained model file

	// create minibatches 
	size_t minibatch_size = MINIBATCH_SIZE;
	std::vector<WordIdSentences> in_s_mono_data_minibatch, in_t_mono_data_minibatch;
	std::vector<std::vector<WordIdSentence> > dev_src_minibatch, dev_trg_minibatch;
	std::vector<size_t> in_s_mono_data_ids_minibatch, in_t_mono_data_ids_minibatch;

	// source
	cerr << endl << "Creating minibatches for in-domain source mono data (using minibatch_size=" << minibatch_size << ")..." << endl;
	create_minibatches(in_mono_data.first, minibatch_size, in_s_mono_data_minibatch);// for in-domain source monolingual data
	in_s_mono_data_ids_minibatch.resize(in_s_mono_data_minibatch.size());// create a sentence list for this train minibatch
	std::iota(in_s_mono_data_ids_minibatch.begin(), in_s_mono_data_ids_minibatch.end(), 0);

	// target
	cerr << "Creating minibatches for in-domain target mono data (using minibatch_size=" << minibatch_size << ")..." << endl;
	create_minibatches(in_mono_data.second, minibatch_size, in_t_mono_data_minibatch);// for in-domain target monolingual data
	in_t_mono_data_ids_minibatch.resize(in_t_mono_data_minibatch.size());// create a sentence list for this train minibatch
	std::iota(in_t_mono_data_ids_minibatch.begin(), in_t_mono_data_ids_minibatch.end(), 0);

	// dev
	cerr << "Creating minibatches for development data (using minibatch_size=" << "1024" /*minibatch_size*/ << ")..." << endl;
	create_minibatches(devel_cor, 1024/*minibatch_size*/, dev_src_minibatch, dev_trg_minibatch);// on dev

	// shuffle minibatches
	cerr << endl << "***SHUFFLE" << endl << endl;
	std::shuffle(in_s_mono_data_ids_minibatch.begin(), in_s_mono_data_ids_minibatch.end(), *dynet::rndeng);
	std::shuffle(in_t_mono_data_ids_minibatch.begin(), in_t_mono_data_ids_minibatch.end(), *dynet::rndeng);

	// get/pre-compute mm scores over in-domain target monolingual data
	if (DEBUG_FLAG) cerr << "Pre-computing \\bar{\\phi} on in-domain target monolingual data..." << endl;
	float score_phi_bar = 0.f;
	unsigned num_samples = 0;
	for (unsigned s = 0; s < in_t_mono_data_ids_minibatch.size() && num_samples < SAMPLING_SIZE; ++s) 
	{
		const auto& tsents = in_t_mono_data_minibatch[in_t_mono_data_ids_minibatch[s]];
		
		dynet::ComputationGraph cg;
		std::vector<float> tmp_scores;
		mm_feas.compute_feature_scores_on_targets(cg, tsents, tmp_scores, true);
		score_phi_bar += tmp_scores[0];
		
		num_samples += tsents.size();	
	}
	score_phi_bar /= num_samples;// averaging
	if (DEBUG_FLAG) cerr << "\\bar{\\phi}=" << score_phi_bar << endl;
	// --- sanity check
	// computing \bar{\phi} on translations for sampled in-domain source sentences produced by initial translation model
	//  FIXME
	if (DEBUG_FLAG) sanity_check(tf, mm_feas, in_s_mono_data_minibatch, in_s_mono_data_ids_minibatch);
	// ---
	 
	// model stats on dev
	//cerr << "Computing mm scores on dev..." << endl;
	transformer::ModelStats dstats(dev_eval_mea/*5: mm measure*/);
	get_dev_stats(devel_cor, tfc, dstats);
	//std::vector<std::vector<float>> v_dev_emp_scores;
	// FIXME
		
	unsigned report_every_i = TREPORT;
	unsigned dev_every_i_reports = DREPORT;

	unsigned cpt = 0;// count of patience

	cerr << endl << "--------------------------------------------------------------------------------------------------------" << endl;
        cerr << "Pre-trained model scores on dev data..." << endl;
	tf.set_dropout(false);
	eval_on_dev(tf, dev_src_minibatch, dev_trg_minibatch, dstats, dev_eval_mea, dev_eval_infer_algo);// batched version (2-3 times faster)
	//eval_mm_on_dev(tf, mm_feas, devel_cor, softmax_temp, v_dev_emp_scores, dstats);
	dstats.update_best_score(cpt);
        cerr << "***DEV: " << "sents=" << devel_cor.size() << " src_unks=" << dstats._words_src_unk << " trg_unks=" << dstats._words_tgt_unk << " " << dstats.get_score_string(true) << endl;
	cerr << "--------------------------------------------------------------------------------------------------------" << endl;	

	unsigned sid = 0, id = 0, last_print = 0;
	MyTimer timer_epoch("completed in"), timer_iteration("completed in");
	unsigned epoch = 0;
	while (epoch < max_epochs) {
		transformer::ModelStats tstats_mm;

		tf.set_dropout(true);// enable dropout

		for (unsigned iter = 0; iter < dev_every_i_reports;) {
			if (id == in_s_mono_data_ids_minibatch.size()) { 
				//timing
				cerr << "***Epoch " << epoch << " is finished. ";
				timer_epoch.show();

				epoch++;

				id = 0;
				sid = 0;
				last_print = 0;

				// learning rate scheduler 1: after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.
				if (lr_epochs > 0 && epoch >= lr_epochs)
					p_sgd->learning_rate /= lr_eta_decay; 

				if (epoch >= max_epochs) break;

				// shuffle the access order
				cerr << "***SHUFFLE" << endl;
				std::shuffle(in_s_mono_data_ids_minibatch.begin(), in_s_mono_data_ids_minibatch.end(), *dynet::rndeng);				

				timer_epoch.reset();
			}

			// build graph for this instance
			dynet::ComputationGraph cg;// dynamic computation graph for each data batch
			if (DEBUGGING_FLAG){//http://dynet.readthedocs.io/en/latest/debugging.html
				cg.set_immediate_compute(true);
				cg.set_check_validity(true);
			}

			// get samples from the current model
			auto& ssents = in_s_mono_data_minibatch[in_s_mono_data_ids_minibatch[id]];

			dynet::Expression i_xent, i_mm, i_xent_mm;
			transformer::ModelStats ctstats_mm;
				
			// pre-computed score for \bar{\phi}
			std::vector<float> v_emp_scores(ssents.size() * mm_feas._num_samples, score_phi_bar);

			// build graph and get model loss
			i_xent = compute_mm_loss(cg, ssents, tf, v_emp_scores, mm_feas, softmax_temp, ctstats_mm, &i_mm);
			dynet::Expression i_xent_sum = dynet::sum_batches(i_xent);

			// total loss with reinforced moment matching scores
			i_xent_mm = dynet::sum_batches(dynet::cmult(i_xent, i_mm));
			
			if (PRINT_GRAPHVIZ) {
				cerr << "***********************************************************************************" << endl;
				cg.print_graphviz();
				cerr << "***********************************************************************************" << endl;
			}

			// perform forward computation for aggregate objective
			//cerr << "forward:" << endl;
			cg.incremental_forward(i_xent_mm);

			// grab the parts of the objective
			float loss = dynet::as_scalar(cg.get_value(i_xent_sum.i));
			if (!is_valid(loss)){
				std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				++id;
				continue;
			}
			//cerr << "loss=" << loss_mm << endl;
			
			// observe other scores
			std::vector<float> v_xent_losses = dynet::as_vector(cg.get_value(i_xent.i));
			if (DEBUG_FLAG){
				cerr << "xent_losses: ";
				for (auto& sc : v_xent_losses)
					cerr << sc << " ";
				cerr << endl;
			}
	
			// collect stats
			tstats_mm._scores[1] += loss;
                        tstats_mm._words_src += ctstats_mm._words_src;
                        tstats_mm._words_src_unk += ctstats_mm._words_src_unk;
                        tstats_mm._words_tgt += ctstats_mm._words_tgt;
                        tstats_mm._words_tgt_unk += ctstats_mm._words_tgt_unk;

			// perform backward
			//cerr << "backward:" << endl;
			cg.backward(i_xent_mm);

			// update parameters
			//cerr << "update params:" << endl;
			p_sgd->update();

			sid += in_s_mono_data_minibatch[in_s_mono_data_ids_minibatch[id]].size();
			iter += in_s_mono_data_minibatch[in_s_mono_data_ids_minibatch[id]].size();
		
			if (sid / report_every_i != last_print 
					|| iter >= dev_every_i_reports
					|| id + 1 == in_s_mono_data_ids_minibatch.size()){
				last_print = sid / report_every_i;

				float elapsed = timer_iteration.elapsed();

				p_sgd->status();
				cerr << "sents=" << sid << " ";
				
				cerr /*<< "loss=" << tstats._scores[1]*/ << "src_unks=" << tstats_mm._words_src_unk << " trg_unks=" << tstats_mm._words_tgt_unk << " " << tstats_mm.get_score_string() << ' ';
                                cerr /*<< "time_elapsed=" << elapsed*/ << "(" << (float)(tstats_mm._words_src + tstats_mm._words_tgt) * 1000.f / elapsed << " words/sec)" << endl;
			}
			   		 
			++id;
		}

		// show score on dev data?
		tf.set_dropout(false);// disable dropout for evaluating dev data

		// sample a random sentence (for observing translations during training progress)
		if (SAMPLING_TRAINING){// Note: this will slow down the training process, suitable for debugging only.
			dynet::ComputationGraph cg;
			WordIdSentence target;// raw translation (w/o scores)			
			cerr << endl << "---------------------------------------------------------------------------------------------------" << endl;
			cerr << "***Source: " << get_sentence(in_s_mono_data_minibatch[in_s_mono_data_ids_minibatch[id]][0], tf.get_source_dict()) << endl;
			tf.sample(cg, in_s_mono_data_minibatch[in_s_mono_data_ids_minibatch[id]][0], target);
			cerr << "***Sampled translation: " << get_sentence(target, tf.get_target_dict()) << endl;
			tf.greedy_decode(cg, in_s_mono_data_minibatch[in_s_mono_data_ids_minibatch[id]][0], target);
			cerr << "***Greedy translation: " << get_sentence(target, tf.get_target_dict()) << endl;
			/*std::vector<WordIdSentence> targets;
			tf.beam_decode(cg, train_src_minibatch[train_ids_minibatch[id]][0], targets, 4, 5);
			for (auto& tgt : targets)
				cerr << "***Beam translation: " << get_sentence(tgt, tf.get_target_dict()) << endl;*/
			cerr << "---------------------------------------------------------------------------------------------------" << endl << endl;

			/* for debugging only
			for (auto& src : train_src_minibatch[train_ids_minibatch[id]])
				cerr << "***Source: " << get_sentence(src, tf.get_source_dict()) << endl;
			tf.sample(cg, train_src_minibatch[train_ids_minibatch[id]], targets);
			for (auto& tgt : targets)
				cerr << "***Sampled translation (batched): " << get_sentence(tgt, tf.get_target_dict()) << endl;
			tf.greedy_decode(cg, train_src_minibatch[train_ids_minibatch[id]], targets);
			for (auto& tgt : targets)
				cerr << "***Greedy translation (batched): " << get_sentence(tgt, tf.get_target_dict()) << endl;

			cerr << "***Source: " << get_sentence(train_src_minibatch[train_ids_minibatch[id]][0], tf.get_source_dict()) << endl;
			std::vector<WordIdSentence> samples;
			std::vector<float> v_probs;
			tf.sample_sentences(cg, train_src_minibatch[train_ids_minibatch[id]][0], 5, samples, v_probs);
			cerr << "***Sampled translations: " << endl;
			for (unsigned id = 0; id < samples.size(); id++){
				auto& sample = samples[id];
				cerr << get_sentence(sample, tf.get_target_dict()) << " (log-probability=" << v_probs[id]/sample.size() << ")" << endl;
			}*/
		}

		timer_iteration.reset();
		
		//eval_on_dev(tf, devel_cor, dstats, dev_eval_mea, dev_eval_infer_algo);// non-batched version
		eval_on_dev(tf, dev_src_minibatch, dev_trg_minibatch, dstats, dev_eval_mea, dev_eval_infer_algo);// batched version (2-3 times faster)
		//eval_mm_on_dev(tf, mm_feas, devel_cor, softmax_temp, v_dev_emp_scores, dstats);
		float elapsed = timer_iteration.elapsed();

		// update best score and save parameter to file
		dstats.update_best_score(cpt);
		if (cpt == 0){
			// FIXME: consider average checkpointing?
			tf.save_params_to_file(params_out_file);
		}

		// verbose
		cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		cerr << "***DEV [epoch=" << (float)epoch + (float)sid/(float)in_mono_data.first.size() << " eta=" << p_sgd->learning_rate << "]" << " sents=" << devel_cor.size() << " src_unks=" << dstats._words_src_unk << " trg_unks=" << dstats._words_tgt_unk << " " << dstats.get_score_string() << ' ';
		if (cpt > 0) cerr << "(not improved, best score on dev so far: " << dstats.get_score_string(false) << ") ";
		cerr << "[completed in " << elapsed << " ms]" << endl;

		// learning rate scheduler 2: if the model has not been improved for lr_patience times, decrease the learning rate by lr_eta_decay factor.
		if (lr_patience > 0 && cpt > 0 && cpt % lr_patience == 0){
			cerr << "The model has not been improved for " << lr_patience << " times. Decreasing the learning rate..." << endl;
			p_sgd->learning_rate /= lr_eta_decay;
		}

		// another early stopping criterion
		if (patience > 0 && cpt >= patience)
		{
			if (RESET_IF_STUCK){
				cerr << "The model seems to get stuck. Resetting now...!" << endl;
				cerr << "Attempting to resume the training..." << endl;			
				// 1) load the previous best model
				cerr << "Loading previous best model..." << endl;
				tf.initialise_params_from_file(params_out_file);
				// 2) some useful tricks:
				sid = 0; id = 0; last_print = 0; cpt = 0;
				// a) reset SGD trainer, switching to Adam instead!
				if (SWITCH_TO_ADAM){ 
					delete p_sgd; p_sgd = 0;
					p_sgd = new dynet::AdamTrainer(tf.get_model_parameters(), 0.001f/*maybe smaller?*/);
					SWITCH_TO_ADAM = false;// do it once!
				}
				// b) use smaller batch size
				if (USE_SMALLER_MINIBATCH){
					cerr << "Creating minibatches for training data (using minibatch_size=" << minibatch_size/2 << ")..." << endl;
					in_s_mono_data_minibatch.clear();
					in_s_mono_data_ids_minibatch.clear();
					create_minibatches(in_mono_data.first, minibatch_size, in_s_mono_data_minibatch);// for in-domain monolingual data
					in_s_mono_data_ids_minibatch.resize(in_s_mono_data_minibatch.size());
					std::iota(in_s_mono_data_ids_minibatch.begin(), in_s_mono_data_ids_minibatch.end(), 0);	

					minibatch_size /= 2;
					report_every_i /= 2;
				}
				// 3) shuffle the training data
				cerr << "***SHUFFLE" << endl;
				std::shuffle(in_s_mono_data_ids_minibatch.begin(), in_s_mono_data_ids_minibatch.end(), *dynet::rndeng);

				NUM_RESETS--;
				if (NUM_RESETS == 0)
					RESET_IF_STUCK = false;// it's right time to stop anyway!
			}
			else{
				cerr << "The model has not been improved for " << patience << " times. Stopping now...!" << endl;
				cerr << "No. of epochs so far: " << epoch << "." << endl;
				cerr << "Best score on dev: " << dstats.get_score_string(false) << endl;
				cerr << "--------------------------------------------------------------------------------------------------------" << endl;

				break;
			}
		}

		timer_iteration.reset();

		cerr << "--------------------------------------------------------------------------------------------------------" << endl;	

	}

	cerr << endl << "***************************" << endl;
	cerr << "Transformer training completed!" << endl;
}
// ---

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
