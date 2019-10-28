"""Generate transposition, deletion, and substitution primes."""
import numpy as np
import unicodedata

from collections import Counter
from itertools import combinations, chain, product
from string import ascii_lowercase
from old20 import old20


LETTERS = set(ascii_lowercase)
VOWELS = {"a", "e", "i", "o", "u", "y", "j"}
CONSONANTS = set(ascii_lowercase) - VOWELS


def strip_accents(s):
    """Strip accents from a unicode string."""
    # FROM: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string # noqa
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


class NGramScorer(object):

    def __init__(self, n):
        """An NGram Scorer."""
        self.n = n

    def fit(self, reference_corpus):
        """Fit the scorer to a corpus."""
        self.gram_dict = self.generate_ngram_counts(reference_corpus)

    def generate_ngram_counts(self, words):
        """Generate counts of ngrams."""
        words = set(words)
        grams = Counter()

        for w in words:
            grams.update(self.ngrams(w, self.n))

        return grams

    def score(self, word):
        """Calculate the mean bigram frequency."""
        grams = list(self.ngrams(word, self.n))
        return sum([self.gram_dict[g] for g in grams]) / len(grams)

    @staticmethod
    def ngrams(x, n):
        """Return all character ngrams without padding."""
        _mask = "#" * (n-1)
        x = f"{_mask}{x}{_mask}"
        for idx in range(0, len(x)-(n-1)):
            yield x[idx:idx+n]


class NeighborhoodScorer(object):

    def __init__(self, k):
        """A neighborhood scorer."""
        self.k = k

    def fit(self, reference_corpus):
        """Fit the scorer on a corpus."""
        self.reference_corpus = reference_corpus

    def score(self, word):
        """Score the word on neighborhood."""
        return old20([word], self.reference_corpus, n=self.k)[0] / self.k


class Trnsps(object):

    def __init__(self,
                 reference_corpus,
                 scorers,
                 allow_outer=False):
        """Reference corpus."""
        if not all([not set(strip_accents(x)) - LETTERS
                    for x in reference_corpus]):
            raise ValueError("Some words were not alphabetical.")
        if not isinstance(scorers, (list, tuple, set)):
            scorers = (scorers,)
        self.reference_corpus = set(reference_corpus)
        self.vocab = set(chain(*reference_corpus))
        self.vowels = VOWELS
        self.consonants = set(ascii_lowercase) - VOWELS
        self.allow_outer = allow_outer
        self.offset = 0 if allow_outer else 1
        # The minimum length for any transformation is 2.
        self.length_limit = 3 if not self.allow_outer else 1
        self.scorers = scorers
        for scorer in self.scorers:
            scorer.fit(self.reference_corpus)

    @staticmethod
    def find_diffs(x, y):
        """Find the differing indices of two words."""
        if len(x) != len(y):
            raise ValueError(f"len({x}) != len({y})")
        return [idx for idx, (x, y) in enumerate(zip(x, y)) if x != y]

    def score(self, word):
        """Give a score to a word."""
        return np.array([x.score(word)
                         for x in self.scorers])

    def _generic_func(self, words, indices, function, k=10):
        """
        A generic function that does something to words.

        This function takes as input a set of words, a set of indices, and
        a function that takes as input a set of indices and a word.

        This transformation is then scored by its mean bigram frequency.

        Parameters
        ----------
        words : list of str
            A list of strings representing the words.
        indices : list of list of integers
            A list of doubly or triply nested list of integers, representing:
            for each word, the possible transformations for that word.
            Whether the list is doubly or triply nested depends on the
            transformation: some transformations require different sets of
            indices.
        function : callable
            The function is always a function that takes as input a word and
            a set of indices or an index, depending on the task.
            The function returns a string which is the transformed word.
        k : int, default 10
            The number of items to return

        Returns
        -------
        items : tuple
            A tuple consisting of the word and its transformations.
            Transformations are represented as a string and a score, where the
            score is the absolute difference in mean bigram frequency between
            the original word and its transformation.

        """
        lengths = np.array([len(w) for w in words])
        assert all([not set(strip_accents(x)) - LETTERS for x in words])
        assert np.all(lengths > self.length_limit)
        for w, index in zip(words, indices):
            res = {}
            score = self.score(w)
            for i in index:
                for new_w in function(w, i):
                    if new_w == w or new_w in self.reference_corpus:
                        continue
                    new_score = self.score(new_w)
                    res[new_w] = np.sum(np.abs(1 - (new_score / score)))
            yield w, sorted(res.items(), key=lambda x: x[1])[:k]

    def transposition(self, words, min_c=0, max_c=2, n=1, k=10):
        """
        Generate substitutions.

        Parameters
        ----------
        words : list of str
            The words for which to generate substitutions.
        min_c : int, optional, default 0
            The minimum distance in letters between transpositions
        max_c : int, optional, default 2
            The maximum distance in letters between transpositions
        n : int
            The number of substitutions to apply to each word.
        k : int
            The number of candidates to return.

        Returns : substitutions
            the substitution neighbors of the words in words.

        """
        assert min_c >= 0 and min_c < max_c

        def index_generator(w_len, n, min_c, max_c, offset):
            combs = combinations(range(offset, w_len-offset), 2)
            combs = [(x, y) for x, y in combs if min_c < abs(x - y) <= max_c]
            return (x for x in combinations(combs, n)
                    if len(set(chain(*x))) == n * 2)

        def sub_trans(word, indices):
            word = list(word)
            for x, y in indices:
                word[x], word[y] = word[y], word[x]
            return ["".join(word)]

        indices = (index_generator(len(w), n, min_c, max_c, self.offset)
                   for w in words)
        return self._generic_func(words, indices, sub_trans, k=k)

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
        def index_generator(w_len, n, offset):
            return combinations(range(offset, w_len-offset), n)

        def sub_subs(w, indices):
            letters = set(w)

            allowed_vowels = VOWELS - letters
            allowed_consonants = CONSONANTS - letters
            classes = []
            for idx in indices:
                if strip_accents(w[idx]) in VOWELS:
                    classes.append(allowed_vowels)
                else:
                    classes.append(allowed_consonants)
            for bundle in product(*classes):
                w_ = list(w)
                for idx, lett in zip(indices, bundle):
                    w_[idx] = lett
                yield "".join(w_)

        if indices is None:
            indices = (index_generator(len(w), n, self.offset) for w in words)
        return self._generic_func(words, indices, sub_subs, k=k)

    def specific_substitution(self, words, indices, k=10):
        """
        Generate substitutions in specific positions.

        Parameters
        ----------
        words : list of str
            The words for which to generate deletions.
        indices : list of indices
            A list, the same length of words, which contains which indices
            to substitute in that specific word.
        k : int
            The number of candidates to return.

        Returns
        -------
        substitutions : list
            the subtitutions neighbors of the words in wordlist.

        """
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

        Returns
        -------
        deletions : list
            The deletion neighbors of the words

        """
        def index_generator(w_len, n, offset):
            return combinations(range(offset, w_len-offset), n)

        def func(w, indices):
            return ["".join([x for idx, x in enumerate(w)
                            if idx not in indices])]

        indices = (index_generator(len(w), n, self.offset) for w in words)
        return self._generic_func(words, indices, func, k=k)

    def insertion(self, words, n=1, k=10):
        """
        Generate insertions.

        Parameters
        ----------
        words : list of str
            The words for which to generate insertions.
        n : int
            The number of insertions to apply to each word.
        k : int
            The number of candidates to return.

        Returns
        -------
        deletions : list
            The insertion neighbors of the words

        """
        def index_generator(w_len, n, offset):
            return combinations(range(offset, w_len-offset), n)

        def func(w, indices):
            # Make sure they are sorted, otherwise insertion fails.
            indices = sorted(indices)
            fragments = []
            prev_idx = 0
            lett = LETTERS - set(w)
            for x in indices:
                fragments.append(w[prev_idx:x])
                prev_idx = x
            if indices[-1] != len(w) - 1:
                fragments.append(w[prev_idx:])
            for bundle in product(*[lett] * len(indices)):
                new_w = []
                for f, l in zip(fragments, bundle):
                    new_w.extend([f, l])
                if indices[-1] != len(w) - 1:
                    new_w.append(fragments[-1])
                yield "".join(new_w)

        indices = (index_generator(len(w), n, self.offset) for w in words)
        return self._generic_func(words, indices, func, k=k)

    def transposition_substitution(self, words, min_c=1, max_c=2, n=1):
        """First generate transpositions, then substitute"""
        t = self.transposition(words, min_c, max_c, n, k=1)
        t = [(x, y[0][0]) for x, y in t if y]
        indices = [(self.find_diffs(x, y),) for x, y in t]

        words, transpositions = zip(*t)
        d = tuple(self.specific_substitution(words, indices))
        _, deletions = zip(*d)
        deletions = tuple([x[0][0] for x in deletions])
        return words, transpositions, deletions
