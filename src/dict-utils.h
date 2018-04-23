#pragma once

#include <memory>
#include <iostream>
#include <sstream>
#include <vector>

#include "dynet/dict.h"

using namespace std;
using namespace dynet;

inline void load_vocabs(const std::string& src_vocab_file, const std::string& trg_vocab_file
	, dynet::Dict& sd, dynet::Dict& td, bool freeze=true);
inline void load_vocab(const std::string& vocab_file
	, dynet::Dict& d, bool freeze=true);
inline void load_joint_vocab(const std::string& vocab_file
	, dynet::Dict& sd, dynet::Dict& td, bool freeze=true);
inline void save_vocabs(const std::string& src_vocab_file, const std::string& trg_vocab_file
	, dynet::Dict& sd, dynet::Dict& td);
inline void save_vocab(const std::string& vocab_file
	, dynet::Dict& d);

inline void load_vocabs(const std::string& src_vocab_file, const std::string& trg_vocab_file
	, dynet::Dict& sd, dynet::Dict& td, bool freeze)
{
	if ("" == src_vocab_file || "" == trg_vocab_file) return;

	cerr << endl << "Loading vocabularies from files..." << endl;
	cerr << "Source vocabulary file: " << src_vocab_file << endl;
	cerr << "Target vocabulary file: " << trg_vocab_file << endl;
	ifstream if_src_vocab(src_vocab_file), if_trg_vocab(trg_vocab_file);
	std::string sword, tword;
	while (getline(if_src_vocab, sword)) sd.convert(sword);
	while (getline(if_trg_vocab, tword)) td.convert(tword);

	// automatically add sentinel markers
	sd.convert("<s>");// source
	sd.convert("</s>");
	sd.convert("<unk>");
	td.convert("<s>");// target
	td.convert("</s>");
	td.convert("<unk>");	

	cerr << "Source vocabluary size: " << sd.size() << endl;
	cerr << "Target vocabluary size: " << td.size() << endl;

	if (freeze){
		sd.freeze();
		td.freeze();
	}
}

inline void load_joint_vocab(const std::string& vocab_file
	, dynet::Dict& sd, dynet::Dict& td, bool freeze)
{
	if ("" == vocab_file) return;

	cerr << "Loading joint source and target vocabulary from file: " << vocab_file << endl;
	ifstream if_vocab(vocab_file);
	std::string word;
	while (getline(if_vocab, word)) sd.convert(word);

	// automatically add sentinel markers
	sd.convert("<s>");
	sd.convert("</s>");
	sd.convert("<unk>");
		
	cerr << "Joint vocabluary size: " << sd.size() << endl;

	if (freeze) sd.freeze();

	td = sd;
}

inline void load_vocab(const std::string& vocab_file
	, dynet::Dict& d, bool freeze)
{
	if ("" == vocab_file) return;

	cerr << endl << "Loading vocabulary from file..." << endl;
	cerr << "Vocabulary file: " << vocab_file << endl;
	ifstream if_vocab(vocab_file);
	std::string word;
	while (getline(if_vocab, word)) d.convert(word);

	// automatically add sentinel markers
	d.convert("<s>");
	d.convert("</s>");
	d.convert("<unk>");
	
	cerr << "Vocabluary size: " << d.size() << endl;

	if (freeze) d.freeze();
}

inline void save_vocabs(const std::string& src_vocab_file, const std::string& trg_vocab_file
	, dynet::Dict& sd, dynet::Dict& td)
{
	if ("" == src_vocab_file || "" == trg_vocab_file) return;

	const auto& swords = sd.get_words();
	const auto& twords = td.get_words();

	ofstream of_svocab(src_vocab_file);
	for (auto& sword : swords)
		of_svocab << sword << endl;
	ofstream of_tvocab(trg_vocab_file);
	for (auto& tword : twords)
		of_tvocab << tword << endl;
}

inline void save_vocab(const std::string& vocab_file
	, dynet::Dict& d)
{
	if ("" == vocab_file) return;

	const auto& words = d.get_words();

	ofstream of_vocab(vocab_file);
	for (auto& word : words)
		of_vocab << word << endl;
}


