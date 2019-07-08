# trnsps

A small tool for generating transposition, deletion, and substitution errors.
For every one of these operations, it is possible to come up with a multitude of alternatives for a given word.
For example, the word `hospital` can be transformed into `hopsital` and `hpsoital` with a single transposition. It is clear, however, that the former obeys the orthotactic rules of english more closely than the second one.
To pick between these alternatives, all operations are returned in rank order, where the rank is decided according to the absolute difference between the mean bigram frequency of the original word and the edited word.
Note that outer letters are *never* transposted, deleted, or substituted.

# Example

```python
from trnsps import transposition

my_corpus = ["godless", "sodless", "respondent", "stockless", "miles"]
result = list(transposition(["godless"], my_corpus))

```

# License

MIT
