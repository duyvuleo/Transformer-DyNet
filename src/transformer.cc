#include "transformer.h"

using namespace std;
using namespace dynet;
using namespace transformer;

int main(int argc, char** argv) {
	cerr << "*** DyNet initialization ***" << endl;
	auto dyparams = dynet::extract_dynet_params(argc, argv);
	dynet::initialize(dyparams);	

	return EXIT_SUCCESS;
}


