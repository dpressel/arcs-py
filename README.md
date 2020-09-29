arcs-py
=======

Python implementation of Arc-Eager and Arc-Hybrid Greedy Dependency Parsing trained with a dynamic oracle described in:

__Goldberg, Yoav, and Joakim Nivre. "Training Deterministic Parsers with Non-Deterministic Oracles." (2013)__

__Goldberg, Yoav, and Joakim Nivre. "A Dynamic Oracle for Arc-Eager Dependency Parsing" (2012)__

The sample data comes from Question Bank and a sample of PTB provided by NLTK in the corpora section, which I converted to a labeled dependency CONLL file using LTH converter and David Vadas' patches.  The sample data also contains non-projective parses, which are ignored by the sample driver program.

If you are looking for a modern flexible dependency parser that supports non-projective parses and is near SoTA, you probably want to use a neural parser architecture like the one in [mead](https://github.com/dpressel/mead-baseline)
