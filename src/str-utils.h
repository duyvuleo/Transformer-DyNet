#pragma once

#include <sstream>
#include <vector>

using namespace std;

std::vector<std::string> split_words(const std::string & str);
std::string print_vector(const std::vector<float>& vec, unsigned len=3);

std::vector<std::string> split_words(const std::string &line) {
	std::istringstream in(line);
	std::string word;
	std::vector<std::string> res;
	while(in) {
		in >> word;
		if (!in || word.empty()) break;
		res.push_back(word);
	}
	return res;
}

std::string print_vector(const std::vector<float>& vec, unsigned len){
	std::stringstream ss;
	unsigned l = 0;
	for (auto val : vec){
		if (l >= len) break;
		if (l == vec.size() - 1)	
			ss << val;
		else
			ss << val << " ";
		++l;
	}
	return ss.str();
}

