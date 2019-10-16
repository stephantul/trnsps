"""Generate transposition, deletion, and substitution primes."""
import numpy as np
import unicodedata

from collections import Counter
from itertools import combinations, chain, product
from string import ascii_lowercase
from functools import partial


LETTERS = set(ascii_lowercase)
VOWELS = {"a", "e", "i", "o", "u", "y", "j"}
CONSONANTS = set(ascii_lowercase) - VOWELS


def strip_accents(s):
    """Strip accents from a unicode string."""
    # FROM: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string # noqa
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


class Trnsps(object):

    def __init__(self, reference_corpus, strategy="log", allow_outer=False):
        """Reference corpus."""
        if not all([not set(strip_accents(x)) - LETTERS
                    for x in reference_corpus]):
            raise ValueError("Some words were not alphabetical.")
        self.reference_corpus = set(reference_corpus)
        self.strategy = strategy
        self.bigrams = self.generate_bigram_counts(self.reference_corpus)
        self.vocab = set(chain(*reference_corpus))
        self.vowels = VOWELS
        self.consonants = set(ascii_lowercase) - VOWELS
        self.allow_outer = allow_outer

    def generate_bigram_counts(self, words, n=2):
        """Generate counts of bigrams."""
        words = set(words)
        grams = Counter()

        for w in words:
            grams.update(self.ngrams(w, n))
        if self.strategy == "log":
            grams = Counter({k: np.log10(v) for k, v in grams.items()})

        return grams

    def mean_bigram_freq(self, word):
        """Calculate the mean bigram frequency."""
        grams = list(self.ngrams(word, 2))
        return sum([self.bigrams[g] for g in grams]) / len(grams)

    @staticmethod
    def ngrams(x, n):
        """Return all character ngrams without padding."""
        if len(x) < n:
            raise ValueError(f"len({x}) < n. Raise your n, or remove word")
        for idx in range(0, len(x)-(n-1)):
            yield x[idx:idx+n]

    @staticmethod
    def find_diffs(x, y):
        """Find the differing indices of two words."""
        if len(x) != len(y):
            raise ValueError(f"len({x}) != len({y})")
        return [idx for idx, (x, y) in enumerate(zip(x, y)) if x != y]

    def generic_func(self, words, indices, function, n=1, k=10):
        lengths = np.array([len(w) for w in words])
        assert all([not set(x) - LETTERS for x in words])
        assert np.all(lengths > 3)
        for w, index in zip(words, indices):
            res = {}
            freq = self.mean_bigram_freq(w)
            for i in index:
                for new_w in function(w, i):
                    if new_w == w or new_w in self.reference_corpus:
                        continue
                    new_freq = self.mean_bigram_freq(new_w)
                    res[new_w] = abs(freq - new_freq)
            yield w, sorted(res.items(), key=lambda x: x[1])[:k]

    def transposition(self, words, min_c=0, max_c=2, n=1, k=10):

        def index_generator(w_len, n, min_c, max_c):
            combs = combinations(range(1, w_len-1), 2)
            combs = [(x, y) for x, y in combs if min_c < abs(x - y) <= max_c]
            return (x for x in combinations(combs, n)
                    if len(set(chain(*x))) == n * 2)

        def sub_trans(word, indices):
            word = list(word)
            for x, y in indices:
                word[x], word[y] = word[y], word[x]
            return ["".join(word)]

        idxgen = partial(index_generator, min_c=min_c, max_c=max_c)
        indices = (idxgen(len(w), n) for w in words)
        return self.generic_func(words, indices, sub_trans, n=n, k=k)

    def substitution(self, words, indices=None, n=1, k=10):
        """
        Generate substitutions.

        Parameters
        ----------
        words : list of str
            The words for which to generate substitutions.
        n : int
            The number of substitutions to apply to each word.
        k : int
            The number of candidates to return.

        Returns : substitutions
            the substitution neighbors of the words in words.

        """
        def index_generator(w_len, n):
            return combinations(range(1, w_len-1), n)

        def sub_subs(word, indices):
            word = list(word)
            letters = set(word)

            allowed_vowels = VOWELS - letters
            allowed_consonants = CONSONANTS - letters
            classes = []
            for idx in indices:
                if strip_accents(word[idx]) in VOWELS:
                    classes.append(allowed_vowels)
                else:
                    classes.append(allowed_consonants)
            for bundle in product(*classes):
                w = list(word)
                for idx, lett in zip(indices, bundle):
                    w[idx] = lett
                yield "".join(w)

        if indices is None:
            indices = (index_generator(len(w), n) for w in words)
        return self.generic_func(words, indices, sub_subs, n=n, k=k)

    def specific_substitution(self, words, indices, k=10):
        assert len(words) == len(indices)
        return self.substitution(words, indices, 1, k)

    def deletion(self, words, n=1, k=10):
        """
        Generate deletions.

        Parameters
        ----------
        words : list of str
            The words for which to generate deletions.
        n : int
            The number of deletions to apply to each word.
        k : int
            The number of candidates to return.

        Returns : deletions
            the deletion neighbors of the words in wordlist.

        """
        def index_generator(w_len, n):
            return combinations(range(1, w_len-1), n)

        def func(w, indices):
            return ["".join([x for idx, x in enumerate(w)
                            if idx not in indices])]

        indices = (index_generator(len(w), n) for w in words)
        return self.generic_func(words, indices, func, n=n, k=k)

    def transposition_substitution(self, words, min_c=1, max_c=2, n=1):
        """First generate transpositions, then substitute"""
        t = self.transposition(words, min_c, max_c, n, k=1)
        t = [(x, y[0][0]) for x, y in t if y]
        indices = [self.find_diffs(x, y) for x, y in t]

        words, transpositions = zip(*t)
        d = tuple(self.specific_substitution(words, indices))
        _, deletions = zip(*d)
        deletions = [x[0][0] for x in deletions]
        return words, transpositions, deletions
