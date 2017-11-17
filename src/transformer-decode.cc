/* This is an implementation of Transformer architecture from https://arxiv.org/abs/1706.03762 (Attention is All You need).
* Developed by Cong Duy Vu Hoang
* Updated: 1 Nov 2017
*/

#include "transformer.h"

using namespace std;
using namespace dynet;
using namespace transformer;

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
		("test,T", value<string>(), "file containing development sentences.")
		("src-vocab", value<string>()->default_value(""), "file containing source vocabulary file; none by default (will be built from train file)")
		("trg-vocab", value<string>()->default_value(""), "file containing target vocabulary file; none by default (will be built from train file)")
		("train-percent", value<unsigned>()->default_value(100), "use <num> percent of sentences in training data; full by default")
		//-----------------------------------------
		("shared-embeddings", "use shared source and target embeddings (in case that source and target use the same vocabulary; none by default")
		//-----------------------------------------
		("dynet-autobatch", value<unsigned>()->default_value(0), "impose the auto-batch mode (support both GPU and CPU); no by default")
		//-----------------------------------------
		("initialise,i", value<string>(), "load initial parameters from file")
		//-----------------------------------------
		("nlayers", value<unsigned>()->default_value(6), "use <num> layers for stacked encoder/decoder layers")
		("num-units,u", value<unsigned>()->default_value(512), "use <num> dimensions for number of units")
		("num-heads,h", value<unsigned>()->default_value(8), "use <num> fors number of heads in multi-head attention mechanism")
		//-----------------------------------------
		("position-encoding", value<unsigned>()->default_value(1), "impose position encoding (0: none; 1: learned positional encoding; 2: sinusoid encoding); 1 by default")
		//-----------------------------------------
		("attention-type", value<unsigned>()->default_value(1), "impose attention type (1: Luong attention type; 2: Bahdanau attention type); 1 by default")
		//-----------------------------------------
		("r2l-target", "use right-to-left direction for target during training; default not")
		//-----------------------------------------
		("swap", "swap roles of source and target, i.e., learn p(source|target)")
		//-----------------------------------------
		("verbose,v", "be extremely chatty")
		//-----------------------------------------
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


	return EXIT_SUCCESS;
}


