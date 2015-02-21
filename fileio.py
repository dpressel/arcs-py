import csv

WORD = 0
POS = 1
HEAD = 2
LABEL = 3


def read_conll_deps(f):

    sentences = []

    with open(f) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')

        sentence = []

        for row in reader:
            if len(row) == 0:
                sentence = [tok if tok[HEAD] is not -1 else (tok[WORD], tok[POS], len(sentence), tok[LABEL]) for tok in sentence]
                sentences.append(sentence)
                sentence = []
                continue
            sentence.append((row[1].lower(), row[3], int(row[6]) - 1, row[7]))

    return sentences
