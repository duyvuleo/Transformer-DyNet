/* This is an implementation of Transformer architecture from https://arxiv.org/abs/1706.03762 (Attention is All You need).
* Developed by Cong Duy Vu Hoang
* Updated: 1 Nov 2017
*/

#include "ensemble-decoder.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <sys/stat.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace dynet;
using namespace transformer;
using namespace boost::program_options;

// ---
bool load_model_config(const std::string& model_cfg_file
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
	, dynet::Dict& sd
	, dynet::Dict& td
	, const transformer::SentinelMarkers& sm);
// ---

// ---
void decode(const std::string test_file
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
	, unsigned beam_size=5
	, unsigned length_ratio=2.f
	, unsigned int lc=0 /*line number to be continued*/
	, bool remove_unk=false /*whether to include <unk> in the output*/
	, bool r2l_target=false /*right-to-left decoding*/);
void decode_nbest(const std::string test_file
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
	, unsigned topk
	, const std::string& nbest_style
	, unsigned beam_size=5
	, unsigned length_ratio=2.f
	, unsigned int lc=0 /*line number to be continued*/
	, bool remove_unk=false /*whether to include <unk> in the output*/
	, bool r2l_target=false /*right-to-left decoding*/);
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
		("dynet-autobatch", value<unsigned>()->default_value(0), "impose the auto-batch mode (support both GPU and CPU); no by default")
		//-----------------------------------------
		("model-path,p", value<std::string>()->default_value("."), "specify pre-trained model path")
		//-----------------------------------------
		("test,T", value<std::string>(), "file containing testing sentences.")
		("lc", value<unsigned int>()->default_value(0), "specify the sentence/line number to be continued (for decoding only); 0 by default")
		//-----------------------------------------
		("beam,b", value<unsigned>()->default_value(1), "size of beam in decoding; 1: greedy by default")
		("alpha,a", value<float>()->default_value(0.6f), "length normalisation hyperparameter; 0.6f by default") // follow the GNMT paper!
		("topk,k", value<unsigned>(), "use <num> top kbest entries; none by default")
		("nbest-style", value<std::string>()->default_value("simple"), "style for nbest translation outputs (moses|simple); simple by default")
		("length-ratio", value<unsigned>()->default_value(2), "target_length = source_length * TARGET_LENGTH_LIMIT_FACTOR; 2 by default")
		//-----------------------------------------
		("remove-unk", "remove <unk> in the output; default not")
		//-----------------------------------------
		("r2l-target", "use right-to-left direction for target during training; default not")
		//-----------------------------------------
		("swap", "swap roles of source and target, i.e., learn p(source|target)")
		//-----------------------------------------
		("verbose,v", "be extremely chatty")
		("dynet-profiling", value<int>()->default_value(0), "enable/disable auto profiling (https://github.com/clab/dynet/pull/1088/commits/bc34db98fa5e2e694f54f0e6b1d720d517c7530e)")// for debugging only		
		//-----------------------------------------
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
		|| !(vm.count("model-path") || !vm.count("test")))
	{
		cout << opts << "\n";
		return EXIT_FAILURE;
	}

	// get and check model path
	std::string model_path = vm["model-path"].as<std::string>();
	struct stat sb;
	if (stat(model_path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))
		cerr << endl << "All model files will be loaded from: " << model_path << "." << endl;
	else
		TRANSFORMER_RUNTIME_ASSERT("The model-path does not exist!");

	// Model recipe
	dynet::Dict sd, td;// vocabularies
	SentinelMarkers sm;// sentinel markers
	std::vector<std::shared_ptr<transformer::TransformerModel>> v_tf_models;

	std::string config_file = model_path + "/model.config";// configuration file path
	if (stat(config_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)){// check existence	
		// load vocabulary from file(s)
		std::string vocab_file = model_path + "/" + "src-tgt.joint-vocab";
		if (stat(vocab_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)){
			load_vocab(vocab_file, sd);
			td = sd;
		}
		else{
			std::string src_vocab_file = model_path + "/" + "src.vocab";
			std::string tgt_vocab_file = model_path + "/" + "tgt.vocab";
			load_vocabs(src_vocab_file, tgt_vocab_file, sd, td);
		}

		transformer::SentinelMarkers sm;
		sm._kSRC_SOS = sd.convert("<s>");
		sm._kSRC_EOS = sd.convert("</s>");
		sm._kTGT_SOS = td.convert("<s>");
		sm._kTGT_EOS = td.convert("</s>");

		// load models
		std::string config_file = model_path + "/model.config";
		if (!load_model_config(config_file, v_tf_models, sd, td, sm))
			TRANSFORMER_RUNTIME_ASSERT("Failed to load model(s)!");
	}
	else TRANSFORMER_RUNTIME_ASSERT("Failed to load model(s) from: " + std::string(model_path) + "!");

	// length normalisation hyperparameter for beam search
	_len_norm_alpha = vm["alpha"].as<float>();

	// input test file
	// the output will be printed to stdout!
	std::string test_input_file = vm["test"].as<std::string>();

	// decode the input file
	if (vm.count("topk"))
		decode_nbest(test_input_file, v_tf_models, vm["topk"].as<unsigned>(), vm["nbest-style"].as<std::string>(), vm["beam"].as<unsigned>(), vm["length-ratio"].as<unsigned>(), vm["lc"].as<unsigned int>(), vm.count("remove-unk"), vm.count("r2l-target"));
	else
		decode(test_input_file, v_tf_models, vm["beam"].as<unsigned>(), vm["length-ratio"].as<unsigned>(), vm["lc"].as<unsigned int>(), vm.count("remove-unk"), vm.count("r2l-target"));

	return EXIT_SUCCESS;
}
//************************************************************************************************************************************************************

// ---
bool load_model_config(const std::string& model_cfg_file
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
	, dynet::Dict& sd
	, dynet::Dict& td
	, const transformer::SentinelMarkers& sm)
{
	cerr << "Loading model(s) from configuration file: " << model_cfg_file << "..." << endl;	

	v_models.clear();

	ifstream inpf(model_cfg_file);
	assert(inpf);
	
	unsigned i = 0;
	std::string line;
	while (getline(inpf, line)){
		if ("" == line) break;

		// each line has the format: 
		// <num-units> <num-heads> <nlayers> <ff-num-units-factor> <encoder-emb-dropout> <encoder-sub-layer-dropout> <decoder-emb-dropout> <decoder-sublayer-dropout> <attention-dropout> <ff-dropout> <use-label-smoothing> <label-smoothing-weight> <position-encoding-type> <position-encoding-flag> <max-seq-len> <attention-type> <ff-activation-type> <use-hybrid-model> <your-trained-model-path>
		// e.g.,
		// 128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 0 300 1 1 0 0 <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu_run1
		// 128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 0 300 1 1 0 0 <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu_run2
		cerr << "Loading model " << i+1 << "..." << endl;
		std::stringstream ss(line);

		transformer::TransformerConfig tfc;
		std::string model_file;

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
		tfc._is_training = false;
		tfc._use_dropout = false;

		v_models.push_back(std::shared_ptr<transformer::TransformerModel>());
		v_models[i].reset(new transformer::TransformerModel(tfc, sd, td));
		cerr << "Model file: " << model_file << endl;
		v_models[i].get()->initialise_params_from_file(model_file);// load pre-trained model from file
		cerr << "Count of model parameters: " << v_models[i].get()->get_model_parameters().parameter_count() << endl;

		i++;
	}

	cerr << "Done!" << endl << endl;

	return true;
}
// ---

// ---
void decode(const std::string test_file
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
	, unsigned beam_size
	, unsigned length_ratio
	, unsigned int lc /*line number to be continued*/
	, bool remove_unk /*whether to include <unk> in the output*/
	, bool r2l_target /*right-to-left decoding*/)
{
	dynet::Dict& sd = v_models[0].get()->get_source_dict();
	dynet::Dict& td = v_models[0].get()->get_target_dict();
	const transformer::SentinelMarkers& sm = v_models[0].get()->get_config()._sm;

	if (beam_size <= 0) TRANSFORMER_RUNTIME_ASSERT("Beam size must be >= 1!");

	EnsembleDecoder ens(td);
	ens.set_beam_size(beam_size);
	ens.set_length_ratio(length_ratio);

	cerr << "Reading test examples from " << test_file << endl;
	ifstream in(test_file);
	assert(in);

	MyTimer timer_dec("completed in");
	std::string line;
	WordIdSentence source;
	unsigned int lno = 0;
	while (std::getline(in, line)) {
		if (lno++ < lc) continue;// continued decoding

		source = dynet::read_sentence(line, sd);

		if (source.front() != sm._kSRC_SOS && source.back() != sm._kSRC_EOS) {
			cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
			abort();
		}

		ComputationGraph cg;// dynamic computation graph
		WordIdSentence target;//, aligns;

		EnsembleDecoderHypPtr trg_hyp = ens.generate(cg, source, v_models);
		if (trg_hyp.get() == nullptr) {
			target.clear();
			//aligns.clear();
		} 
		else {
			target = trg_hyp->get_sentence();
			//aligns = trg_hyp->get_alignment();
		}

		if (r2l_target)
			std::reverse(target.begin() + 1, target.end() - 1);

		bool first = true;
		for (auto &w: target) {
			if (!first) cout << " ";

			if (remove_unk && w == sm._kTGT_UNK) continue;

			cout << td.convert(w);

			first = false;
		}
		cout << endl;

		//break;//for debug only
	}

	double elapsed = timer_dec.elapsed();
	cerr << "Decoding is finished!" << endl;
	cerr << "Decoded " << (lno - lc) << " sentences, completed in " << elapsed/1000 << "(s)" << endl;
}
// ---

// ---
void decode_nbest(const std::string test_file
	, std::vector<std::shared_ptr<transformer::TransformerModel>>& v_models
	, unsigned topk
	, const std::string& nbest_style
	, unsigned beam_size
	, unsigned length_ratio
	, unsigned int lc /*line number to be continued*/
	, bool remove_unk /*whether to include <unk> in the output*/
	, bool r2l_target /*right-to-left decoding*/)
{
	dynet::Dict& sd = v_models[0].get()->get_source_dict();
	dynet::Dict& td = v_models[0].get()->get_target_dict();
	const transformer::SentinelMarkers& sm = v_models[0].get()->get_config()._sm;

	if (topk < 1) TRANSFORMER_RUNTIME_ASSERT("topk must be >= 1!");

	if (beam_size <= 0) TRANSFORMER_RUNTIME_ASSERT("Beam size must be >= 1!");

	EnsembleDecoder ens(td);
	ens.set_beam_size(beam_size);
	ens.set_length_ratio(length_ratio);

	cerr << "Reading test examples from " << test_file << endl;
	ifstream in(test_file);
	assert(in);

	MyTimer timer_dec("completed in");
	std::string line;
	WordIdSentence source;
	unsigned int lno = 0;
	while (std::getline(in, line)) {
		if (lno++ < lc) continue;// continued decoding

		source = dynet::read_sentence(line, sd);

		if (source.front() != sm._kSRC_SOS && source.back() != sm._kSRC_EOS) {
			cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
			abort();
		}

		ComputationGraph cg;// dynamic computation graph
		WordIdSentence target;//, aligns;
		float score = 0.f;

		std::vector<EnsembleDecoderHypPtr> v_trg_hyps = ens.generate_nbest(cg, source, v_models, topk);
		for (auto& trg_hyp : v_trg_hyps){
			if (trg_hyp.get() == nullptr) {
				target.clear();
				//aligns.clear();
			} 
			else {
				target = trg_hyp->get_sentence();
				score = trg_hyp->get_score();
				//aligns = trg_hyp->get_alignment();
			}

			if (target.size() < 2) continue;// <=2, e.g., <s> ... </s>?
		
			if (r2l_target)
		   		std::reverse(target.begin() + 1, target.end() - 1);

			if (nbest_style == "moses"){
				// n-best with Moses's format 
				// <line_number1> ||| source ||| target1 ||| TransformerModelScore=score1 || score1
				// <line_number2> ||| source ||| target2 ||| TransformerModelScore=score2 || score2
				//...

				// follows Moses's nbest file format
				std::stringstream ss;

				// source text
				ss /*<< lno << " ||| "*/ << line << " ||| ";
			   
				// target text
				bool first = true;
				for (auto &w: target) {
					if (!first) ss << " ";
					ss << td.convert(w);
					first = false;
				}
		
				// score
				ss << " ||| " << "TransformerModelScore=" << -score / (target.size() - 1) << " ||| " << -score / (target.size() - 1);//normalized by target length, following Moses's N-best format.
		
				ss << endl;

				cout << ss.str();
			}
			else if (nbest_style == "simple"){
				// simple format with target1 ||| target2 ||| ...
				std::stringstream ss;
				bool first = true;
				for (auto &w: target) {
					if (!first) ss << " ";
					ss << td.convert(w);
					first = false;
				}
				ss << " ||| ";
				cout << ss.str();
			}
			else TRANSFORMER_RUNTIME_ASSERT("Unknown style for nbest translation outputs!");
		}

		if (nbest_style == "simple") cout << endl;

		//break;//for debug only
	}

	double elapsed = timer_dec.elapsed();
	cerr << "Decoding is finished!" << endl;
	cerr << "Decoded " << (lno - lc) << " sentences, completed in " << elapsed/1000 << "(s)" << endl;
}
// ---

