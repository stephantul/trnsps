"""Generate transposition, deletion, and substitution primes."""
import numpy as np
import unicodedata

from collections import Counter
from itertools import combinations, chain
from string import ascii_lowercase
from itertools import product


LETTERS = set(ascii_lowercase)
VOWELS = {"a", "e", "i", "o", "u", "y", "j"}
CONSONANTS = set(ascii_lowercase) - VOWELS


def strip_accents(s):
    """Strip accents from a unicode string."""
    # FROM: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string # noqa
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


class Trnsps(object):

    def __init__(self, reference_corpus, strategy="log"):
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

    @staticmethod
    def ngrams_position(word, indices):
        """Get bigrams from a specific position in a word."""
        a = []
        for x in indices:
            a.extend([word[x-1:x+1], word[x-2:x]])
        return a

    def position_ngram_freq(self, word, indices):
        """Get the frequency of bigrams at a specific position."""
        return sum([self.bigrams[x] for x
                    in self.ngrams_position(word, indices)])

    def substitution(self, words, n=1, k=10):
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
            the substitution neighbors of the words in wordlist.

        """
        lengths = np.array([len(w) for w in words])
        assert all([not set(x) - LETTERS for x in words])
        assert np.all(lengths > 3)
        for w in words:
            # Generate all possible transpositions.
            res = {}
            for indices in combinations(range(1, len(w)-1), n):
                freq = self.position_ngram_freq(w, indices)
                for new_word in self._sub_subloop(w, indices):
                    new_freq = self.position_ngram_freq(new_word, indices)
                    res[new_word] = abs(freq - new_freq)
            yield w, sorted(res.items(), key=lambda x: x[1])[:k]

    def specific_substitution(self,
                              words,
                              indices,
                              k=10):
        """Generate substitutions for specific positions."""
        assert all([not set(x) - LETTERS for x in words])
        assert len(words) == len(indices)

        for w, i in zip(words, indices):

            freq = self.position_ngram_freq(w, i)
            res = {}
            for new_word in self._sub_subloop(w, i):
                new_freq = self.position_ngram_freq(new_word, i)
                res[new_word] = abs(freq - new_freq)
            yield w, sorted(res.items(), key=lambda x: x[1])[:k]

    def _sub_subloop(self, word, indices):
        """Shared code between the specific and general substitution."""
        cv_grid = []
        lett_in_word = set(strip_accents(word))
        vow = self.vowels - lett_in_word
        cons = self.consonants - lett_in_word
        cv_grid = [vow if word[x] in self.vowels else cons for x in indices]

        for bundle in product(*cv_grid):
            pw = list(word)
            for idx, lett in zip(indices, bundle):
                pw[idx] = lett
            pw = "".join(pw)
            if pw in self.reference_corpus or pw == word:
                continue
            yield pw

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
        lengths = np.array([len(w) for w in words])
        assert np.all(lengths > 3)
        assert all([not set(x) - LETTERS for x in words])
        for w in words:
            freq = self.mean_bigram_freq(w)
            # Generate all possible transpositions.
            res = {}
            for indices in combinations(range(1, len(w)-1), n):
                pw = list(w)
                indices = set(indices)
                pw = "".join([x for idx, x in enumerate(pw)
                              if idx not in indices])
                if pw == w or pw in self.reference_corpus:
                    continue
                new_freq = self.position_ngram_freq(pw, indices)
                res[pw] = abs(new_freq - freq)

            yield w, list(sorted(res.items(), key=lambda x: x[1]))[:k]

    def transposition(self, words, constraint=2, n=1, k=10):
        """
        Generate transpositions.

        Parameters
        ----------
        words : list of str
            The words for which to generate transpositions.
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
        assert all([not set(x) - LETTERS for x in words])

        for w in words:
            base = self.mean_bigram_freq(w)
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
                if pw == w or pw in self.reference_corpus:
                    continue
                res[pw] = abs(self.mean_bigram_freq(pw) - base)

            yield w, list(sorted(res.items(), key=lambda x: x[1]))[:k]

    def transposition_substitution(self, words, constraint=2, n=1):
        """First generate transpositions, then substitute"""
        t = self.transposition(words, constraint, n, k=1)
        t = [(x, y[0][0]) for x, y in t if y]
        indices = [self.find_diffs(x, y) for x, y in t]

        words, transpositions = zip(*t)
        d = tuple(self.specific_substitution(words, indices))
        _, deletions = zip(*d)
        deletions = [x[0][0] for x in deletions]
        return words, transpositions, deletions
