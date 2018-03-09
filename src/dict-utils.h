#pragma once

#include <memory>
#include <iostream>
#include <sstream>
#include <vector>

#include "dynet/dict.h"

using namespace std;
using namespace dynet;

inline void load_vocabs(const string& src_vocab_file, const string& trg_vocab_file
	, dynet::Dict& sd, dynet::Dict& td, bool freeze=true);

inline void load_vocab(const string& vocab_file
	, dynet::Dict& d, bool freeze=true);

inline void load_vocabs(const string& src_vocab_file, const string& trg_vocab_file
	, dynet::Dict& sd, dynet::Dict& td, bool freeze)
{
	if ("" == src_vocab_file || "" == trg_vocab_file) return;

	cerr << endl << "Loading vocabularies from files..." << endl;
	cerr << "Source vocabulary file: " << src_vocab_file << endl;
	cerr << "Target vocabulary file: " << trg_vocab_file << endl;
	ifstream if_src_vocab(src_vocab_file), if_trg_vocab(trg_vocab_file);
	string sword, tword;
	while (getline(if_src_vocab, sword)) sd.convert(sword);
	while (getline(if_trg_vocab, tword)) td.convert(tword);
	
	cerr << "Source vocabluary size: " << sd.size() << endl;
	cerr << "Target vocabluary size: " << td.size() << endl;

	if (freeze){
		sd.freeze();
		td.freeze();
	}
}

inline void load_vocab(const string& vocab_file
	, dynet::Dict& d, bool freeze)
{
	if ("" == vocab_file) return;

	cerr << endl << "Loading vocabulary from file..." << endl;
	cerr << "Vocabulary file: " << vocab_file << endl;
	ifstream if_vocab(vocab_file);
	string word;
	while (getline(if_vocab, word)) d.convert(word);
	
	cerr << "Vocabluary size: " << d.size() << endl;

	if (freeze) d.freeze();
}


