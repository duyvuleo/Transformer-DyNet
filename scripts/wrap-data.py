# coding=utf-8

import sys
import os
import collections
import itertools

def threshold_vocab(fname, threshold):
	word_counts = collections.Counter()
	with open(fname) as fin:
		for line in fin:
			for token in line.split():
				word_counts[token] += 1

	ok = set()
	for word, count in sorted(word_counts.items()):
		if count >= threshold:
			ok.add(word)
	return ok

def load_vocab_from_file(fname):
	vocab = set()
	fv = open(fname, 'rb')
	for line in fv:
		vocab.add(line.strip())
	return vocab

argc = len(sys.argv)

if argc == 1 or (argc == 2 and (sys.argv[1] == "--help" or sys.argv[1] == "-h")): # print help
	# python scripts/wrap-data.py <src-lang-id> <trg-lang-id> <train-prefix> <dev-prefix> <test-prefix> <vocab-prefix>
	print "Usage 1: python scripts/wrap-data.py <src-lang-id> <trg-lang-id> <train-prefix> <dev-prefix> <test-prefix> <vocab-prefix>"
	print "Usage 2: python scripts/wrap-data.py <src-lang-id> <trg-lang-id> <train-prefix> <dev-prefix> <test-prefix> <vocab-prefix>"
	exit()

if argc == 7 or argc == 8:
	sfname = sys.argv[3] + "." + sys.argv[1] # e.g., '/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/train.en'
	tfname = sys.argv[3] + "." + sys.argv[2] # e.g., '/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/train.vi'

	if argc == 7:
		source_vocab = load_vocab_from_file(sys.argv[6] + "." + sys.argv[1]) # e.g., '/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/vocab.en'
		target_vocab = load_vocab_from_file(sys.argv[6] + "." + sys.argv[2]) # e.g., '/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/vocab.vi'
	else: 
		source_vocab = threshold_vocab(sfname, int(sys.argv[6]))
		target_vocab = threshold_vocab(tfname, int(sys.argv[7]))
elif argc == 5:
	if os.path.exists(sys.argv[4]):
		vocab = load_vocab_from_file(sys.argv[4])
		ftail = "vb"
	else: 
		vocab = threshold_vocab(sys.argv[1], int(sys.argv[4]))
		ftail = "f" + sys.argv[4]
else: exit()

def process_corpus(sf, tf, of, sv, tv):
	with open(of, 'w') as fout:
		with open(sf) as sin:
			with open(tf) as tin:
				for sline, tline in itertools.izip(sin, tin):
					print >>fout, '<s>',
					for token in sline.split():
						if token in sv:
							print >>fout, token,
						else:
							print >>fout, '<unk>',
					print >>fout, '</s>', '|||',

					print >>fout, '<s>',
					for token in tline.split():
						if token in tv:
							print >>fout, token,
						else:
							print >>fout, '<unk>',
					print >>fout, '</s>'

def process_corpus_r(sf, tf, of, sv, tv):
	with open(of, 'w') as fout:
		with open(sf) as sin:
			with open(tf) as tin:
				for sline, tline in itertools.izip(sin, tin):
					print >>fout, '<s>',
					for token in tline.split():
						if token in tv:
							print >>fout, token,
						else:
							print >>fout, '<unk>',
					print >>fout, '</s>', '|||',

					print >>fout, '<s>',
					for token in sline.split():
						if token in sv:
							print >>fout, token,
						else:
							print >>fout, '<unk>',
					print >>fout, '</s>'

def process_corpus_mono(sf, of, v):
	with open(of, 'w') as fout:
		with open(sf) as sin:
			for line in sin:
				print >>fout, '<s>',
				for token in line.split():
					if token in v:
						print >>fout, token,
					else:
						print >>fout, '<unk>',
				print >>fout, '</s>'

def process_test(sf, of, vocab):
	with open(of, 'w') as fout:
		with open(sf) as sin:
				for sline in sin:
					print >>fout, '<s>',
					for token in sline.split():
						if token in vocab:
							print >>fout, token,
						else:
							print >>fout, '<unk>',
					print >>fout, '</s>'

if argc == 7 or argc == 8: # translation task
	ofname = sys.argv[3] + "." + sys.argv[1] + "-" + sys.argv[2] + ".capped" # '/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/train.en-vi.vcb.capped'

	process_corpus(sfname, tfname, ofname, source_vocab, target_vocab) #train (for training)
	process_corpus(sys.argv[4] + "." + sys.argv[1], sys.argv[4] + "." + sys.argv[2], sys.argv[4] + "." + sys.argv[1] + "-" + sys.argv[2] + ".capped", source_vocab, target_vocab) # e.g., process_corpus('/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/tst2012.en', '/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/tst2012.vi', '/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/tst2012.en-vi.vcb.capped') #dev (for training)
	process_test(sys.argv[4] + "." + sys.argv[1], sys.argv[4] + "." + sys.argv[1] + ".capped", source_vocab) # e.g., process_test('/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/tst2012.en','/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/tst2012.en.vcb.capped', source_vocab) #dev (for decoding)
	process_test(sys.argv[5] + "." + sys.argv[1], sys.argv[5] + "." + sys.argv[1] + ".capped", source_vocab) # e.g., process_test('/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/tst2013.en','/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/tst2013.en.vcb.capped', source_vocab) #test (for decoding)
elif argc == 5: # language modeling task	
	process_corpus_mono(sys.argv[1], sys.argv[1] + "." + ftail + ".capped", vocab) #train (for training)
	process_test(sys.argv[2], sys.argv[2] + "." + ftail + ".capped", vocab)
	process_test(sys.argv[3], sys.argv[3] + "." + ftail + ".capped", vocab)
	


