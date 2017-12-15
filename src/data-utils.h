#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "dynet/dict.h"

using namespace std;
using namespace dynet;

WordIdCorpus read_corpus(const string &filename
	, dynet::Dict* sd, dynet::Dict* td
	, bool cid=true/*corpus id, 1:train;0:otherwise*/
	, unsigned slen=0, bool r2l_target=false
	, bool swap=false);

WordIdCorpus read_corpus(const string &filename
	, dynet::Dict* sd, dynet::Dict* td
	, bool cid
	, unsigned slen, bool r2l_target
	, bool swap)
{
	int kSRC_SOS = sd->convert("<s>");
	int kSRC_EOS = sd->convert("</s>");
	int kTGT_SOS = td->convert("<s>");
	int kTGT_EOS = td->convert("</s>");

	ifstream in(filename);
	assert(in);

	WordIdCorpus corpus;

	string line;
	int lc = 0, stoks = 0, ttoks = 0;
	while (getline(in, line)) {
		WordIdSentence source, target;

		if (!swap)
			read_sentence_pair(line, source, *sd, target, *td);
		else read_sentence_pair(line, source, *td, target, *sd);

		// reverse the target if required
		if (r2l_target) 
			std::reverse(target.begin() + 1/*BOS*/,target.end() - 1/*EOS*/);

		// constrain sentence length(s)
		if (cid/*train only*/ && slen > 0/*length limit*/){
			if (source.size() > slen || target.size() > slen)
				continue;// ignore this sentence
		}

		if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
				(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
			stringstream ss;
			ss << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
			assert(ss.str().c_str());

			abort();
		}

		if (source.size() < 3 || target.size() < 3){ // ignore empty sentences, e.g., <s> </s>
			continue;
		}

		corpus.push_back(WordIdSentencePair(source, target));

		stoks += source.size();
		ttoks += target.size();

		++lc;
	}

	// print stats
	if (cid)
		cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd->size() << " & " << td->size() << " types\n";
	else 
		cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t)\n" ;

	return corpus;
}
