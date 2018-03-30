MTEval
==============

`MTEval` - Collection of evaluation metrics and algorithms for machine translation.
It has been integrated inside Transformer-DyNet. 

About
-----

https://github.com/odashi/mteval

List of Metrics
---------------

* **BLEU**
    * Identifier: `BLEU`
    * Parameters:
        * `ngram`: maximum n-gram length (default: `4`)
        * `smooth`: additional counts for >1-gram (default: `0`)
    * Statistics:
        * `len:hyp`: number of words in hypothesis sentences.
        * `len:ref`: number of words in reference sentences.
        * `ngram:%d:hyp`: number of n-grams in the hypothesis sentence.
        * `ngram:%d:match`: number of matched n-grams.
        * `samples`: number of evaluation samples.

* **NIST**
    * Identifier: `NIST`
    * Parameters:
        * `ngram`: maximum n-gram length (default: `5`)
    * Statistics:
        * `len:hyp`: number of words in hypothesis sentences.
        * `len:ref`: number of words in reference sentences.
        * `ngram:%d:hyp`: number of n-grams in the hypothesis sentence.
        * `ngram:%d:match`: cumulative weighted n-gram matches.
        * `samples`: number of evaluation samples.

* **RIBES**
    * Identifier: `RIBES`
    * Parameters:
        * `alpha`: weight of unigram precision (default: `0.25`)
        * `beta`: weight of brevity penalty (default: `0.1`)
    * Statistics:
        * `brevity`: cumulative brevity penalty for each evaluation sample.
        * `nkt`: cumulative Kendall's tau for each evaluation sample.
        * `prec`: cumulative unigram precision for each evaluation sample.
        * `samples`: number of evaluation samples.
        * `score`: cumulative RIBES score for each evaluation sample.

* **Word Error Rate**
    * Identifier: `WER`
    * Parameters:
        * `substitute`: weight of substituting ref/hyp words (default: `1.0`)
        * `insert`: weight of inserting a hyp word (default: `1.0`)
        * `delete`: weight of deleting a hyp word (default: `1.0`)
    * Statistics:
        * `distance`: cumulative Levenshtein distance for each evaluation sample.
        * `samples`: number of evaluation samples.
        * `score`: cumulative WER score for eac

Creators
--------

* Yusuke Oda (@odashi) - Most coding

Contact
-------

Y.Oda
* @odashi_t on Twitter (faster than E-Mail)
* yus.takara (at) gmail.com

