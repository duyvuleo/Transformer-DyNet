/* This is an implementation of Transformer architecture from https://arxiv.org/abs/1706.03762 (Attention is All You need).
* Developed by Cong Duy Vu Hoang
* Updated: 1 Nov 2017
*/

#include "transformer.h"

using namespace std;
using namespace dynet;
using namespace transformer;

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace dynet;
using namespace boost::program_options;
using namespace transformer;

// hyper-paramaters for training
unsigned MINIBATCH_SIZE = 1;

bool DEBUGGING_FLAG = false;

unsigned TREPORT = 50;
unsigned DREPORT = 5000;

bool VERBOSE = false;

// ---
bool load_data(const variables_map& vm
	, WordIdCorpus& train_cor, WordIdCorpus& devel_cor
	, dynet::Dict& sd, dynet::Dict& td
	, SentinelMarkers& sm);
// ---

// ---
dynet::Trainer* create_sgd_trainer(const variables_map& vm, dynet::ParameterCollection& model);
// ---

// ---
void run_train(transformer::TransformerModel &tf, WordIdCorpus &train_cor, WordIdCorpus &devel_cor, 
	Trainer &sgd, 
	string params_out_file, 
	unsigned max_epochs, unsigned patience, 
	unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience);// support batching
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
		("config,c", value<string>(), "config file specifying additional command line options")
		//-----------------------------------------
		("train,t", value<vector<string>>(), "file containing training sentences, with each line consisting of source ||| target.")		
		("devel,d", value<string>(), "file containing development sentences.")
		("slen_limit", value<unsigned>()->default_value(0), "limit the sentence length (either source or target); none by default")
		("src-vocab", value<string>()->default_value(""), "file containing source vocabulary file; none by default (will be built from train file)")
		("trg-vocab", value<string>()->default_value(""), "file containing target vocabulary file; none by default (will be built from train file)")
		("train-percent", value<unsigned>()->default_value(100), "use <num> percent of sentences in training data; full by default")
		//-----------------------------------------
		("shared-embeddings", "use shared source and target embeddings (in case that source and target use the same vocabulary; none by default")
		//-----------------------------------------
		("minibatch-size", value<unsigned>()->default_value(1), "impose the minibatch size for training (support both GPU and CPU); single batch by default")
		("dynet-autobatch", "impose the auto-batch mode (support both GPU and CPU); no by default")
		//-----------------------------------------
		("sgd-trainer", value<unsigned>()->default_value(0), "use specific SGD trainer (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp; 6: cyclical SGD)")
		("sparse-updates", value<bool>()->default_value(true), "enable/disable sparse update(s) for lookup parameter(s); true by default")
		("grad-clip-threshold", value<float>()->default_value(5.f), "use specific gradient clipping threshold (https://arxiv.org/pdf/1211.5063.pdf); 5 by default")
		//-----------------------------------------
		("initialise,i", value<string>(), "load initial parameters from file")
		("parameters,p", value<string>(), "save best parameters to this file")
		//-----------------------------------------
		("nlayers", value<unsigned>()->default_value(6), "use <num> layers for stacked encoder/decoder layers")
		("num-units,u", value<unsigned>()->default_value(512), "use <num> dimensions for number of units")
		("num-heads,h", value<unsigned>()->default_value(8), "use <num> fors number of heads in multi-head attention mechanism")
		//-----------------------------------------
		("dropout-rate", value<float>()->default_value(0.1f), "use dropout thoroughly; 0.1 by default")
		//-----------------------------------------
		("use-label-smoothing", "use label smoothing for cross entropy; no by default")
		("label-smoothing-weight", value<float>()->default_value(0.9f), "use dropout thoroughly; 0.1 by default")
		//-----------------------------------------
		("position-encoding", value<unsigned>()->default_value(1), "impose position encoding (0: none; 1: learned positional encoding; 2: sinusoid encoding); 1 by default")
		("max-seq-len", value<unsigned>()->default_value(500), "impose max sequence length; 500 by default")
		//-----------------------------------------
		("attention-type", value<unsigned>()->default_value(0), "impose attention type (0: Luong attention type; 1: Bahdanau attention type); 0 by default")
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
		("r2l-target", "use right-to-left direction for target during training; default not")
		//-----------------------------------------
		("swap", "swap roles of source and target, i.e., learn p(source|target)")
		//-----------------------------------------
		("treport", value<unsigned>()->default_value(50), "no. of training instances for reporting current model status on training data")
		("dreport", value<unsigned>()->default_value(5000), "no. of training instances for reporting current model status on development data (dreport = N * treport)")
		//-----------------------------------------
		("verbose,v", "be extremely chatty")
		//-----------------------------------------
		("debug", "enable/disable simpler debugging by immediate computing mode or checking validity (refers to http://dynet.readthedocs.io/en/latest/debugging.html)")// for CPU only
	;
	
	store(parse_command_line(argc, argv, opts), vm); 
	if (vm.count("config") > 0)
	{
		ifstream config(vm["config"].as<string>().c_str());
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
		|| vm.count("train") != 1
		|| vm.count("devel") != 1)
	{
		cout << opts << "\n";
		return EXIT_FAILURE;
	}

	// hyper-parameters for training
	DEBUGGING_FLAG = vm.count("debug");
	VERBOSE = vm.count("verbose");
	TREPORT = vm["treport"].as<unsigned>(); 
	DREPORT = vm["dreport"].as<unsigned>(); 
	if (DREPORT % TREPORT != 0) assert("dreport must be divisible by treport.");// to ensure the reporting on development data
	MINIBATCH_SIZE = vm["minibatch_size"].as<unsigned>();

	// load fixed vocabularies from files if required
	dynet::Dict sd, td;

	load_vocabs(vm["src-vocab"].as<string>(), vm["trg-vocab"].as<string>(), sd, td);

	SentinelMarkers sm;
	sm._kSRC_SOS = sd.convert("<s>");
	sm._kSRC_EOS = sd.convert("</s>");
	sm._kTGT_SOS = td.convert("<s>");
	sm._kTGT_EOS = td.convert("</s>");

	// load data files
	WordIdCorpus train_cor, devel_cor;
	if (!load_data(vm, train_cor, devel_cor, sd, td, sm))
		assert("Failed to load data files!");

	// learning rate scheduler
	unsigned lr_epochs = vm["lr_epochs"].as<unsigned>(), lr_patience = vm["lr_patience"].as<unsigned>();
	if (lr_epochs > 0 && lr_patience > 0)
		cerr << "[WARNING] - Conflict on learning rate scheduler; use either lr_epochs or lr_patience!" << endl;

	// transformer configuration
	TransformerConfig tfc(sd.size(), td.size()
		, vm["num-units"].as<unsigned>()
		, vm["num-heads"].as<unsigned>()
		, vm["nlayers"].as<unsigned>()
		, vm["dropout-rate"].as<float>()
		, vm.count("use-label-smoothing")
		, vm["label-smoothing-weight"].as<float>()
		, vm["position-encoding"].as<unsigned>()
		, vm["max-seq-len"].as<unsigned>()
		, sm
		, vm["attention-type"].as<unsigned>());

	// initialise transformer object
	transformer::TransformerModel tf(tfc, sd, td);
	if (vm.count("initialise")) tf.initialise_params_from_file(vm["initialise"].as<string>());// load pre-trained model (for incremental training)

	// create SGD trainer
	Trainer* p_sgd_trainer = create_sgd_trainer(vm, tf.get_model_parameters());

	// train transformer model
	run_train(tf
		, train_cor, devel_cor
		, *p_sgd_trainer
		, vm["parameters"].as<string>() /*best saved model parameter file*/
		, vm["epochs"].as<unsigned>(), vm["patience"].as<unsigned>() /*early stopping*/
		, lr_epochs, vm["lr_eta_decay"].as<float>(), lr_patience)/*learning rate scheduler*/;

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

	vector<string> train_paths = vm["train"].as<vector<string>>();// to handle multiple training data
	if (train_paths.size() > 2) assert("Invalid -t or --train parameter. Only maximum 2 training corpora provided!");	
	//cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
	//train_cor = read_corpus(vm["train"].as<string>(), doco, true, vm["slen_limit"].as<unsigned>(), r2l_target & !swap, vm["eos_padding"].as<unsigned>());
	cerr << endl << "Reading training data from " << train_paths[0] << "...\n";
	train_cor = read_corpus(train_paths[0], &sd, &td, true, vm["slen-limit"].as<unsigned>(), r2l_target & !swap);
	if ("" == vm["src-vocab"].as<string>() 
		&& "" == vm["trg-vocab"].as<string>()) // if not using external vocabularies
	{
		sd.freeze(); // no new word types allowed
		td.freeze(); // no new word types allowed
	}
	if (train_paths.size() == 2)// incremental training
	{
		train_cor.clear();// use the next training corpus instead!	
		cerr << "Reading extra training data from " << train_paths[1] << "...\n";
		train_cor = read_corpus(train_paths[1], &sd, &td, true/*for training*/, vm["slen-limit"].as<unsigned>(), r2l_target & !swap);
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

	if (vm.count("devel")) {
		cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
		devel_cor = read_corpus(vm["devel"].as<string>(), &sd, &td, false/*for development*/, 0, r2l_target & !swap);
	}

	// set up <s>, </s>, <unk> ids
	sd.set_unk("<unk>");
	sm._kSRC_UNK = sd.get_unk_id();
	td.set_unk("<unk>");
	sm._kTGT_UNK = td.get_unk_id();

	if (swap) {
		cerr << "Swapping role of source and target\n";
		std::swap(sd, td);
		std::swap(sm._kSRC_SOS, sm._kTGT_SOS);
		std::swap(sm._kSRC_EOS, sm._kTGT_EOS);
		std::swap(sm._kSRC_UNK, sm._kTGT_UNK);

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
		sgd = new MomentumSGDTrainer(model, vm["lr_eta"].as<float>());
	else if (sgd_type == 2)
		sgd = new AdagradTrainer(model, vm["lr_eta"].as<float>());
	else if (sgd_type == 3)
		sgd = new AdadeltaTrainer(model);
	else if (sgd_type == 4)
		sgd = new AdamTrainer(model, vm["lr_eta"].as<float>());
	else if (sgd_type == 5)
		sgd = new RMSPropTrainer(model, vm["lr_eta"].as<float>());
	else if (sgd_type == 0)//Vanilla SGD trainer
		sgd = new SimpleSGDTrainer(model, vm["lr_eta"].as<float>());
	else
	   	assert("Unknown SGD trainer type! (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp)");
	sgd->clip_threshold = vm["grad-clip-threshold"].as<float>();// * MINIBATCH_SIZE;// use larger gradient clipping threshold if training with mini-batching, correct?
	sgd->sparse_updates_enabled = vm["sparse-updates"].as<bool>();
	if (!sgd->sparse_updates_enabled)
		cerr << "Sparse updates for lookup parameter(s) to be disabled!" << endl;

	return sgd;
}
// ---

// ---
void run_train(transformer::TransformerModel &tf, WordIdCorpus &train_cor, WordIdCorpus &devel_cor, 
	Trainer &sgd, 
	string params_out_file, 
	unsigned max_epochs, unsigned patience, 
	unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience)
{
	// Create minibatches
	vector<vector<WordIdSentence> > train_src_minibatch;
	vector<vector<WordIdSentence> > train_trg_minibatch;
	vector<size_t> train_ids_minibatch, dev_ids_minibatch;
	size_t minibatch_size = MINIBATCH_SIZE;
	create_minibatches(train_cor, minibatch_size, train_src_minibatch, train_trg_minibatch, train_ids_minibatch);
  
	double best_loss = 9e+99;
	
	unsigned report_every_i = TREPORT;//50;
	unsigned dev_every_i_reports = DREPORT;//500; 

	// shuffle minibatches
	cerr << endl << "***SHUFFLE\n";
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

				// Shuffle the access order
				cerr << "***SHUFFLE\n";
				std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

				timer_epoch.reset();
			}

			// build graph for this instance
			ComputationGraph cg;// dynamic computation graph for each data batch
			if (DEBUGGING_FLAG){//http://dynet.readthedocs.io/en/latest/debugging.html
				cg.set_immediate_compute(true);
				cg.set_check_validity(true);
			}
	
			Expression i_cov_penalty, i_fertility_nll;
			transformer::ModelStats ctstats;
			Expression i_xent = tf.build_graph(cg, train_src_minibatch[train_ids_minibatch[id]], train_trg_minibatch[train_ids_minibatch[id]], ctstats);

			Expression i_objective = i_xent;

			// perform forward computation for aggregate objective
			cg.forward(i_objective);

			// grab the parts of the objective
			float closs = as_scalar(cg.get_value(i_xent.i));
			if (!is_valid(closs)){
				std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				++id;
				continue;
			}

			tstats._loss += closs;
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
				cerr /*<< "loss=" << tstats._loss*/ << "src_unks=" << tstats._words_src_unk << " trg_unks=" << tstats._words_tgt_unk << " E=" << (tstats._loss / tstats._words_tgt) << " ppl=" << exp(tstats._loss / tstats._words_tgt) << ' ';
				cerr /*<< "time_elapsed=" << elapsed*/ << "(" << (tstats._words_src + tstats._words_tgt) * 1000 / elapsed << " words/sec)" << endl;  			
			}
			   		 
			++id;
		}

		timer_iteration.reset();

		// show score on dev data?
		tf.set_dropout(false);// disable dropout for evaluating dev data

		transformer::ModelStats dstats;
		for (unsigned i = 0; i < devel_cor.size(); ++i) {
			WordIdSentence ssent, tsent;
			tie(ssent, tsent) = devel_cor[i];  

			ComputationGraph cg;
			auto i_xent = tf.build_graph(cg, WordIdSentences(1, ssent), WordIdSentences(1, tsent), dstats);
			dstats._loss += as_scalar(cg.forward(i_xent));
		}
		
		if (dstats._loss < best_loss) {
			best_loss = dstats._loss;
			tf.save_params_to_file(params_out_file);// FIXME: save params from last K runs?
			cpt = 0;
		}
		else cpt++;

		cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		cerr << "***DEV [epoch=" << (float)epoch + (float)sid/(float)train_cor.size() << " eta=" << sgd.learning_rate << "]" << " sents=" << devel_cor.size() << " src_unks=" << dstats._words_src_unk << " trg_unks=" << dstats._words_tgt_unk << " E=" << (dstats._loss / dstats._words_tgt) << " ppl=" << exp(dstats._loss / dstats._words_tgt) << ' ';
		if (cpt > 0) cerr << "(not improved, best ppl on dev so far = " << exp(best_loss / dstats._words_tgt) << ") ";
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
			cerr << "Best ppl on dev: " << exp(best_loss / dstats._words_tgt) << endl;
			cerr << "--------------------------------------------------------------------------------------------------------" << endl;
			break;
		}
		cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		timer_iteration.reset();
	}

	cerr << endl << "Transformer training completed!" << endl;
}
// ---


