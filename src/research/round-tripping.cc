/*
 * This is an implementation of the following work:
 * Dual Learning for Machine Translation
 * Yingce Xia, Di He, Tao Qin, Liwei Wang, Nenghai Yu, Tie-Yan Liu, Wei-Ying Ma
 * https://papers.nips.cc/paper/6469-dual-learning-for-machine-translation.pdf
 * Developed by Cong Duy Vu Hoang (vhoang2@student.unimelb.edu.au)
 * Date: 21 May 2017
 *
*/

// We call this framework as "round tripping" instead of "dual learning". 

#include "../transformer.h" // transformer
#include "../transformer-lm.h" // transformer-based lm

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace dynet;
using namespace boost::program_options;

unsigned MAX_EPOCH = 10;
unsigned DEV_ROUND = 25000;

bool VERBOSE;

int main_body(variables_map vm);

typedef WordIdSentences MonoData;

// read the data
MonoData read_mono_data(const string &filename, const dynet::Dict& d);

// main round tripping function
void run_round_tripping(transformer::TransformerModel& s2t_mod, transformer::TransformerModel& t2t_mod
		, const MonoData& mono_s, const MonoData& mono_t
		, const WordIdCorpus& dev_cor /*for evaluation*/
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
		("model-path-s2t,p", value<std::string>()->default_value("."), "pre-trained path for the source-to-target transformer model will be loaded from this folder")
		("model-path-t2s,p", value<std::string>()->default_value("."), "pre-trained path for the target-to-source transformer model will be loaded from this folder")
		("model-path-s,p", value<std::string>()->default_value("."), "pre-trained path for the target language model will be loaded from this folder")
		("model-path-t,p", value<std::string>()->default_value("."), "pre-trained path for the source language model will be loaded from this folder")
		("dev,d", value<string>(), "file containing development parallel sentences, with "
			"each line consisting of source ||| target.")
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
	float alpha = vm["alpha"].as<unsigned>();
	float gamma_1 = vm["gamma_1"].as<unsigned>();
	float gamma_2 = vm["gamma_2"].as<unsigned>();
	MAX_EPOCH = vm["epoch"].as<unsigned>();
	DEV_ROUND = vm["dev_round"].as<unsigned>();
	
	//--- load data
	dynet::Dict sdict, tdict;
	
	// monolingual corpora
	// Assume that these monolingual corpora use the same vocabularies with parallel corpus	used for training.
	MonoData mono_cor_s, mono_cor_t;
	cerr << "Reading monolingual source data from " << vm["mono_s"].as<string>() << "...\n";
	mono_cor_s = read_mono_data(vm["mono_s"].as<string>(), sdict);
	cerr << "Reading monolingual target data from " << vm["mono_t"].as<string>() << "...\n";
	mono_cor_t = read_mono_data(vm["mono_t"].as<string>(), tdict);
	
	//--- load models
	// FIXME


	//--- execute dual-learning
	// FIXME
	
	// finished!
	return EXIT_SUCCESS;
}

void run_round_tripping(transformer::TransformerModel& s2t_mod, transformer::TransformerModel& t2t_mod
		, const MonoData& mono_s, const MonoData& mono_t
		, const WordIdCorpus& dev_cor /*for evaluation*/
		, unsigned K, unsigned beam_size, float alpha, float gamma_1, float gamma_2 /*hyper-parameters of round tripping framework*/
		, unsigned opt_type)
{
	// FIXME
}

MonoData read_mono_data(const string &filename, const dynet::Dict& d)
{
	MonoData corpus;
	// FIXME
	return corpus;
}




