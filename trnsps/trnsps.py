"""Generate transposition, deletion, and substitution primes."""
import numpy as np
import unicodedata

from collections import Counter
from itertools import combinations, chain
from string import ascii_lowercase
from itertools import product


VOWELS = {"a", "e", "i", "o", "u", "y", "j"}
CONSONANTS = set(ascii_lowercase) - VOWELS


def strip_accents(s):
    """Strip accents from a unicode string."""
    # FROM: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string # noqa
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def generate_bigram_counts(words, n=2):
    """Generate counts of bigrams."""
    words = set(words)
    grams = Counter()

    for w in words:
        grams.update(ngrams(w, n))

    return Counter({k: np.log10(v) for k, v in grams.items()})


def mean_bigram_freq(word, gram_freq):
    """Calculate the mean bigram frequency."""
    return sum([gram_freq[g] for g in ngrams(word, 2)]) / (len(word) - 1)


def ngrams(x, n):
    """Return all character ngrams without padding."""
    for idx in range(0, len(x)-(n-1)):
        yield x[idx:idx+n]


def find_diffs(x, y):
    """Find the differing indices of two words."""
    return [idx for idx, (x, y) in enumerate(zip(x, y)) if x != y]


def specific_substitution(words, reference_corpus, indices, k=10):
    """Generate substitutions for specific positions."""
    reference_corpus = set(reference_corpus)
    grams = generate_bigram_counts(reference_corpus)
    vocab = set(chain.from_iterable(reference_corpus))
    res = []
    for w, i in zip(words, indices):
        base = mean_bigram_freq(w, grams)
        res = {}
        for pw in _sub_subloop(w, i, reference_corpus, vocab):
            f = mean_bigram_freq(pw, grams)
            res[pw] = abs(f - base)

        yield w, list(sorted(res.items(), key=lambda x: x[1]))[:k]


def _sub_subloop(word, indices, reference_corpus, vocab):
    cv_grid = []
    vocab = vocab - set(word)
    vow = [x for x in vocab if strip_accents(x) in VOWELS]
    cons = list(vocab - set(vow))
    for x in indices:
        adding = vow if word[x] in VOWELS else cons
        cv_grid.append(adding)
    for bundle in product(*cv_grid):
        pw = list(word)
        for idx, lett in zip(indices, bundle):
            pw[idx] = lett
        pw = "".join(pw)
        if pw == word or pw in reference_corpus:
            continue
        yield pw


def substitution(words, reference_corpus, n=1, k=10):
    """
    Generate deletions.

    Parameters
    ----------
    words : list of str
        The words for which to generate substitutions.
    reference_corpus : list of str
        The reference corpus to use as background knowledge
    n : int
        The number of substitutions to apply to each word.
    k : int
        The number of candidates to return.

    Returns : substitutions
        the substitution neighbors of the words in wordlist.

    """
    lengths = np.array([len(w) for w in words])
    assert np.all(lengths > 3)
    reference_corpus = set(reference_corpus)
    grams = generate_bigram_counts(reference_corpus)
    vocab = set(chain.from_iterable(reference_corpus))
    for w in words:
        base = mean_bigram_freq(w, grams)
        # Generate all possible transpositions.
        res = {}
        for indices in combinations(range(1, len(w)-1), n):
            res.update({w: abs(mean_bigram_freq(w, grams) - base)
                        for w in _sub_subloop(w,
                                              indices,
                                              reference_corpus,
                                              vocab)})
        yield w, list(sorted(res.items(), key=lambda x: x[1]))[:k]


def deletion(words, reference_corpus, n=1, k=10):
    """
    Generate deletions.

    Parameters
    ----------
    words : list of str
        The words for which to generate deletions.
    reference_corpus : list of str
        The reference corpus to use as background knowledge
    n : int
        The number of deletions to apply to each word.
    k : int
        The number of candidates to return.

    Returns : deletions
        the deletion neighbors of the words in wordlist.

    """
    lengths = np.array([len(w) for w in words])
    assert np.all(lengths > 3)
    reference_corpus = set(reference_corpus)
    grams = generate_bigram_counts(reference_corpus)
    for w in words:
        base = mean_bigram_freq(w, grams)
        # Generate all possible transpositions.
        res = {}
        for indices in combinations(range(1, len(w)-1), n):
            pw = list(w)
            indices = set(indices)
            pw = "".join([x for idx, x in enumerate(pw) if idx not in indices])
            if pw == w or pw in reference_corpus:
                continue
            res[pw] = abs(mean_bigram_freq(pw, grams) - base)

        yield w, list(sorted(res.items(), key=lambda x: x[1]))[:k]


def transposition(words, reference_corpus, constraint=2, n=1, k=10):
    """
    Generate transpositions.

    Parameters
    ----------
    words : list of str
        The words for which to generate transpositions.
    reference_corpus : list of str
        The reference corpus to use as background knowledge.
    constraint : int
        The maximum number of intervening letters between two letters.
    n : int
        The number of transposition operations to apply to each word.
    k : int
        The number of candidates to return.

    Returns
    -------
    transpositions : list of str
        The transpositions for the given words.
    """
    lengths = np.array([len(w) for w in words])
    assert np.all(lengths > 3)
    reference_corpus = set(reference_corpus)
    grams = generate_bigram_counts(reference_corpus)

    for w in words:
        base = mean_bigram_freq(w, grams)
        # Generate all possible transpositions.
        res = {}
        combs = combinations(range(1, len(w)-1), 2)
        combs = [(x, y) for x, y in combs if abs(x - y) <= constraint]
        for c in combinations(combs, n):
            # We need to have unique combinations
            if len(set(chain(*c))) != n * 2:
                continue
            pw = list(w)
            for x, y in c:
                pw[x], pw[y] = pw[y], pw[x]
            pw = "".join(pw)
            if pw == w or pw in reference_corpus:
                continue
            res[pw] = abs(mean_bigram_freq(pw, grams) - base)

        yield w, list(sorted(res.items(), key=lambda x: x[1]))[:k]
