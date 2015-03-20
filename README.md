arcs-py
=======

Python implementation of Arc-Eager and Arc-Hybrid Greedy Dependency Parsing trained with a dynamic oracle described in:

__Goldberg, Yoav, and Joakim Nivre. "Training Deterministic Parsers with Non-Deterministic Oracles." (2013)__

__Goldberg, Yoav, and Joakim Nivre. "A Dynamic Oracle for Arc-Eager Dependency Parsing" (2012)__

The sample data comes from Question Bank and a sample of PTB provided by NLTK in the corpora section, which I converted to a labeled dependency CONLL file using LTH converter and David Vadas' patches.  Somebody I hope to get around to making the parser labeled, but for right now its unlabeled only.  The sample data also contains non-projective parses, which are ignored by the sample driver program.
