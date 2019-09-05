import random
from collections import defaultdict


class Configuration:
    def __init__(self, buf, s):
        self.arcs = []
        self.buffer = buf
        self.stack = []
        self.sentence = s


class GoldConfiguration:
    def __init__(self):
        self.heads = {}
        self.deps = defaultdict(lambda: [])


class Classifier:
    def __init__(self, weights, labels):
        self.weights = weights
        self.labels = labels

    def score(self, fv):

        scores = dict((label, 0) for label in self.labels)

        for k, v in fv.items():

            if v == 0:
                continue
            if k not in self.weights:
                continue

            wv = self.weights[k]

            for label, weight in wv.items():
                scores[label] += weight * v

        return scores


class GreedyDepParser:

    SHIFT = 0
    RIGHT = 1
    LEFT = 2
    REDUCE = 3

    MAX_EX_ITER = 5
    MAX_EX_THRESH = 0.8

    def __init__(self, m, feature_extractor):
        self.model = m
        self.fx = feature_extractor
        self.transition_funcs = {}
        self.train_tick = 0
        self.train_last_tick = defaultdict(lambda: 0)
        self.train_totals = defaultdict(lambda: 0)

    def initial(self, sentence):
        pass

    @staticmethod
    def terminal(conf):
        return len(conf.stack) == 0 and len(conf.buffer) == 1

    def legal(self, conf):
        pass

    LUT = ["SHIFT", 'RIGHT', 'LEFT', 'REDUCE']

    def update(self, truth, guess, features):
        def update_feature_label(label, fj, v):
            wv = 0

            try:
                wv = self.model.weights[fj][label]
            except KeyError:
                if fj not in self.model.weights:
                    self.model.weights[fj] = {}
                self.model.weights[fj][label] = 0

            t_delt = self.train_tick - self.train_last_tick[(fj, label)]
            self.train_totals[(fj, label)] += t_delt * wv
            self.model.weights[fj][label] += v
            self.train_last_tick[(fj, label)] = self.train_tick

        self.train_tick += 1
        for f in features.items():
            update_feature_label(truth, f[0], 1.0)
            update_feature_label(guess, f[0], -1.0)

    def dyn_oracle(self, gold_conf, conf, legal_transitions):
        pass

    def avg_weights(self):
        for fj in self.model.weights:
            for label in self.model.weights[fj]:
                total = self.train_totals[(fj, label)]
                t_delt = self.train_tick - self.train_last_tick[(fj, label)]
                total += t_delt * self.model.weights[fj][label]
                avg = round(total / float(self.train_tick))
                if avg:
                    self.model.weights[fj][label] = avg

    @staticmethod
    def get_gold_conf(sentence):
        gold_conf = GoldConfiguration()
        for dep in range(len(sentence)):
            head = sentence[dep][2]
            gold_conf.heads[dep] = head
            if head not in gold_conf.deps:
                gold_conf.deps[head] = []
            gold_conf.deps[head].append(dep)

        return gold_conf

    def run(self, sentence):
        conf = self.initial(sentence)
        while not GreedyDepParser.terminal(conf):
            legal_transitions = self.legal(conf)
            features = self.fx(conf)
            scores = self.model.score(features)
            t_p = max(legal_transitions, key=lambda p: scores[p])
            conf = self.transition(t_p, conf)

        return conf.arcs

    # We need to have arcs that are dominated with no crossing lines, excluding the root
    @staticmethod
    def non_projective(conf):
        for dep1 in conf.heads.keys():
            head1 = conf.heads[dep1]
            for dep2 in conf.heads.keys():
                head2 = conf.heads[dep2]
                if head1 < 0 or head2 < 0:
                    continue
                if (dep1 > head2 and dep1 < dep2 and head1 < head2) or (dep1 < head2 and dep1 > dep2 and head1 < dep2):
                    return True

                if dep1 < head1 and head1 is not head2:
                    if (head1 > head2 and head1 < dep2 and dep1 < head2) or (head1 < head2 and head1 > dep2 and dep1 < dep2):
                        return True
        return False

    def train(self, sentence, iter_num):
        conf = self.initial(sentence)
        gold_conf = GreedyDepParser.get_gold_conf(sentence)
        train_correct = train_all = 0

        n = 0
        while not GreedyDepParser.terminal(conf):
            n += 1
            legal_transitions = self.legal(conf)
            # print('LEGAL ', ' '.join([self.LUT[p] for p in legal_transitions]))
            features = self.fx(conf)
            scores = self.model.score(features)
            t_p = max(legal_transitions, key=lambda p: scores[p])
            zero_cost = self.dyn_oracle(gold_conf, conf, legal_transitions)
            # print(str(n) + ' [ ' + ' '.join([self.LUT[z] for z in zero_cost]) + ' ]')

            if len(zero_cost) == 0:
                raise Exception('no zero cost')

            if t_p not in zero_cost:
                t_o = max(zero_cost, key=lambda p: scores[p])
                self.update(t_o, t_p, features)
                self.explore(t_o, t_p, conf, iter_num)

            else:
                train_correct += 1
                conf = self.transition(t_p, conf)

            train_all += 1
        return train_correct, train_all

    def explore(self, t_o, t_p, conf, iter_i):

        if iter_i > GreedyDepParser.MAX_EX_ITER and random.random() > GreedyDepParser.MAX_EX_THRESH:
            return self.transition(t_p, conf)

        return self.transition(t_o, conf)

    def transition(self, t_p, conf):
        return self.transition_funcs[t_p](conf)


class ArcEagerDepParser(GreedyDepParser):

    def __init__(self, m, f):
        GreedyDepParser.__init__(self, m, f)
        self.transition_funcs[ArcEagerDepParser.SHIFT] = ArcEagerDepParser.shift
        self.transition_funcs[ArcEagerDepParser.RIGHT] = ArcEagerDepParser.arc_right
        self.transition_funcs[ArcEagerDepParser.LEFT] = ArcEagerDepParser.arc_left
        self.transition_funcs[ArcEagerDepParser.REDUCE] = ArcEagerDepParser.reduce

    def initial(self, sentence):
        return Configuration(list(range(len(sentence))) + [len(sentence)], sentence)

    def legal(self, conf):
        """
        Legal transitions for arc-eager dependency parsing
        :param conf: The current state
        :return: any legal transitions
        """
        transitions = [
            GreedyDepParser.SHIFT,
            GreedyDepParser.RIGHT,
            GreedyDepParser.LEFT,
            GreedyDepParser.REDUCE
            ]
        shift_ok = True
        right_ok = True
        left_ok = True
        reduce_ok = True

        if len(conf.buffer) == 1:
            right_ok = shift_ok = False

        if len(conf.stack) == 0:
            left_ok = right_ok = reduce_ok = False
        else:
            s = conf.stack[-1]

            # if the s is already a dependent, we cannot left-arc
            if len(list(filter(lambda hd: s == hd[1], conf.arcs))) > 0:
                left_ok = False
            else:
                reduce_ok = False

        ok = [shift_ok, right_ok, left_ok, reduce_ok]

        legal_transitions = []
        for it in range(len(transitions)):
            if ok[it] is True:
                legal_transitions.append(it)

        return legal_transitions

    def dyn_oracle(self, gold_conf, conf, legal_transitions):
        options = []
        if GreedyDepParser.SHIFT in legal_transitions and ArcEagerDepParser.zero_cost_shift(conf, gold_conf):
            options.append(GreedyDepParser.SHIFT)
        if GreedyDepParser.RIGHT in legal_transitions and ArcEagerDepParser.zero_cost_right(conf, gold_conf):
            options.append(GreedyDepParser.RIGHT)
        if GreedyDepParser.LEFT in legal_transitions and ArcEagerDepParser.zero_cost_left(conf, gold_conf):
            options.append(GreedyDepParser.LEFT)
        if GreedyDepParser.REDUCE in legal_transitions and ArcEagerDepParser.zero_cost_reduce(conf, gold_conf):
            options.append(GreedyDepParser.REDUCE)

        return options

    @staticmethod
    def zero_cost_shift(conf, gold_conf):
        """
        Is a shift zero cost?
        Moving b onto stack means that b will not be able to acquire any head or dependents in S.  Cost
        is number of gold arcs of form (k, b) or (b, k) such that k in S

        :param conf: Working config
        :param gold_conf: Gold config
        :return: Is the cost zero
        """
        if len(conf.buffer) <= 1:
            return False
        b = conf.buffer[0]

        for si in conf.stack:
            if gold_conf.heads[si] == b or (gold_conf.heads[b] == si):
                return False
        return True

    @staticmethod
    def zero_cost_right(conf, gold_conf):
        """
        Adding the arc (s, b) and pushing b onto the stack means that b will not be able to acquire any head in
        S or B, nor any dependents in S.  The cost is the number of gold arcs of form (k, b) such that k in S or B,
        (b, k) such that k in S and no arc (x, k) in working conf.  Cost zero for (s, b) in gold arcs but also
        where s is not the gold head of b but the real head not in S or B and no gold dependents of b in S.
        We return a boolean to identify if right-arc will be zero cost

        :param conf: working configuration (A_c)
        :param gold_conf: gold configuration
        :return: True if zero-cost, false otherwise
        """

        if len(conf.stack) is 0 or len(conf.buffer) is 0:
            return False

        # Stack top
        s = conf.stack[-1]
        # Buffer top
        b = conf.buffer[0]

        # (k, b)
        k = b in gold_conf.heads and gold_conf.heads[b] or -1

        # (s, b) in gold
        if k == s:
            return True

        # (k, b) and k in S or B
        k_b_costs = k in conf.stack or k in conf.buffer

        # (h, d) => k_heads[d] = h
        k_heads = dict((arc[1], arc[0]) for arc in conf.arcs)

        # (b, k)
        b_deps = gold_conf.deps[b]

        # (b, k) and k in S
        b_k_in_stack = list(filter(lambda dep: dep in conf.stack, b_deps))
        b_k_final = list(filter(lambda dep: dep not in k_heads, b_k_in_stack))

        # s is not gold head but real head (k) not in stack or buffer
        # and no gold deps of b in S -- (b, k) doesnt exist on stack
        if k not in conf.buffer and k not in conf.stack and len(b_k_in_stack) is 0:
            return True

        if k_b_costs:
            return False

        return len(b_k_final) == 0

    @staticmethod
    def zero_cost_left(conf, gold_conf):
        """
        Is the cost of a left arc going to be zero?  Adding the arc (b, s) and popping s from the stack
        means that s will not be able to acquire any head or dependents in B.  The cost is the number of gold_arcs
        (k, s) or (s, k) where k in B.

        Cost of the arc found in the gold_arcs is 0, as well as the case where b is not the gold head, but the
        real head is not in B.

        :param conf: The working configuration
        :param gold_conf: The gold arcs
        :return: True if a left-arc would be zero-cost, False otherwise
        """
        if len(conf.stack) is 0 or len(conf.buffer) is 0:
            return False

        s = conf.stack[-1]
        b = conf.buffer[0]

        for bi in range(b, len(conf.sentence) + 1):
            if bi in gold_conf.heads and gold_conf.heads[bi] == s:
                return False
            if b is not bi and gold_conf.heads[s] == bi:
                return False
        return True

    @staticmethod
    def zero_cost_reduce(conf, gold_conf):
        if len(conf.stack) is 0 or len(conf.buffer) is 0:
            return False

        s = conf.stack[-1]
        b = conf.buffer[0]
        for bi in range(b, len(conf.sentence) + 1):
            if bi in gold_conf.heads and gold_conf.heads[bi] == s:
                return False
        return True

    @staticmethod
    def shift(conf):
        b = conf.buffer[0]
        del conf.buffer[0]
        conf.stack.append(b)
        return conf

    @staticmethod
    def arc_right(conf):
        s = conf.stack[-1]
        b = conf.buffer[0]
        del conf.buffer[0]
        conf.stack.append(b)
        conf.arcs.append((s, b))
        return conf

    @staticmethod
    def arc_left(conf):
        #  pop the top off the stack, link the arc, from the buffer
        s = conf.stack.pop()
        b = conf.buffer[0]
        conf.arcs.append((b, s))
        return conf

    @staticmethod
    def reduce(conf):
        conf.stack.pop()
        return conf


class ArcHybridDepParser(GreedyDepParser):

    def __init__(self, m, f):
        GreedyDepParser.__init__(self, m, f)
        self.transition_funcs[ArcHybridDepParser.SHIFT] = ArcHybridDepParser.shift
        self.transition_funcs[ArcHybridDepParser.RIGHT] = ArcHybridDepParser.arc_right
        self.transition_funcs[ArcHybridDepParser.LEFT] = ArcHybridDepParser.arc_left
        self.root = None

    def initial(self, sentence):
        self.root = len(sentence)
        return Configuration(list(range(len(sentence))) + [len(sentence)], sentence)
        # return Configuration([self.root] + range(len(sentence)), sentence)

    def legal(self, conf):
        transitions = []
        left_ok = right_ok = shift_ok = True

        if len(conf.stack) < 2:
            right_ok = False
        if len(conf.stack) == 0 or conf.stack[-1] == self.root:
            left_ok = False

        if shift_ok is True:
            transitions.append(GreedyDepParser.SHIFT)
        if right_ok is True:
            transitions.append(GreedyDepParser.RIGHT)
        if left_ok is True:
            transitions.append(GreedyDepParser.LEFT)
        return transitions

    @staticmethod
    def zero_cost_right(conf, gold_conf):
        """
        Adding the arc (s1, s0) and popping s0 from the stack means that s0 will not be able
        to acquire heads or deps from B.  The cost is the number of arcs in gold_conf of the form
        (s0, d) and (h, s0) where h, d in B.  For non-zero cost moves, we are looking simply for
        (s0, b) or (b, s0) for all b in B
        :param conf:
        :param gold_conf:
        :return:
        """
        s0 = conf.stack[-1]
        for b in conf.buffer:
            if (b in gold_conf.heads and gold_conf.heads[b] is s0) or gold_conf.heads[s0] is b:
                return False
        return True


    @staticmethod
    def zero_cost_left(conf, gold_conf):
        """
        Adding the arc (b, s0) and popping s0 from the stack means that s0 will not be able to acquire
        heads from H = {s1} U B and will not be able to acquire dependents from B U b, therefore the cost is
        the number of arcs in T of form (s0, d) or (h, s0), h in H, d in D

        To have cost, then, only one instance must occur

        :param conf:
        :param gold_conf:
        :return:
        """

        s0 = conf.stack[-1]
        s1 = len(conf.stack) > 2 and conf.stack[-2] or None

        if gold_conf.deps[s0] in conf.buffer:
            return False

        H = conf.buffer[1:] + [s1]
        if gold_conf.heads[s0] in H:
            return False
        return True

    @staticmethod
    def zero_cost_shift(conf, gold_conf):
        """
        Pushing b onto the stack means that b will not be able to acquire
        heads from H = {s1} U S and will not be able to acquire deps from
        D = {s0, s1} U S
        :param conf:
        :param gold_conf:
        :return:
        """
        if len(conf.buffer) < 1:
            return False
        if len(conf.stack) == 0:
            return True

        b = conf.buffer[0]
        # Cost is the number of arcs in T of the form (s0, d) and (h, s0) for h in H and d in D
        if b in gold_conf.heads and gold_conf.heads[b] in conf.stack[0:-1]:
            return False
        ll = len(list(filter(lambda dep: dep in conf.stack, gold_conf.deps[b])))
        return ll == 0

    @staticmethod
    def shift(conf):
        b = conf.buffer[0]
        del conf.buffer[0]
        conf.stack.append(b)
        return conf

    @staticmethod
    def arc_right(conf):
        s0 = conf.stack.pop()
        s1 = conf.stack[-1]
        conf.arcs.append((s1, s0))
        return conf

    @staticmethod
    def arc_left(conf):
        #  pop the top off the stack, link the arc, from the buffer
        s0 = conf.stack.pop()
        b = conf.buffer[0]
        conf.arcs.append((b, s0))
        return conf

    def dyn_oracle(self, gold_conf, conf, legal_transitions):
        options = []
        if GreedyDepParser.SHIFT in legal_transitions and ArcHybridDepParser.zero_cost_shift(conf, gold_conf):
            options.append(GreedyDepParser.SHIFT)
        if GreedyDepParser.RIGHT in legal_transitions and ArcHybridDepParser.zero_cost_right(conf, gold_conf):
            options.append(GreedyDepParser.RIGHT)
        if GreedyDepParser.LEFT in legal_transitions and ArcHybridDepParser.zero_cost_left(conf, gold_conf):
            options.append(GreedyDepParser.LEFT)
        return options

if __name__ == '__main__':

    import argparse
    import fileio
    import fx
    
    parser = argparse.ArgumentParser(description="Sample program showing training and testing dependency parsers")
    parser.add_argument('--parser', help='Parser type (eager|hybrid) (default: eager)', default='eager')
    parser.add_argument('--train', help='CONLL training file', required=True)
    parser.add_argument('--test', help='CONLL testing file', required=True)
    parser.add_argument('--fx', help='Feature extractor', default='ex')
    parser.add_argument('--n', help='Number of passes over training data', default=15, type=int)
    parser.add_argument('-v', action='store_true')
    opts = parser.parse_args()
    
    def filter_non_projective(gold):
            gold_proj = []
            for s in gold:
                gold_conf = GreedyDepParser.get_gold_conf(s)
                if GreedyDepParser.non_projective(gold_conf) is False:
                    gold_proj.append(s)
                elif opts.v is True:
                    print('Skipping non-projective sentence', s)
            return gold_proj

    # Defaults
    feature_extractor = fx.ex
    Parser = ArcEagerDepParser
    
    if opts.fx == 'baseline':
        print('Selecting baseline feature extractor')
        feature_extractor = fx.ex

    
    if opts.parser == 'hybrid':
        print('Using arc-hybrid parser')
        Parser = ArcHybridDepParser

    gold = filter_non_projective(fileio.read_conll_deps(opts.train))
    model = Classifier({}, [0, 1, 2, 3])
    
    parser = Parser(model, feature_extractor)
    print('performing %d iterations' % opts.n)
    for i in range(0, opts.n):
        correct_iter = 0
        all_iter = 0
        random.shuffle(gold)
        for gold_sent in gold:
            correct_s, all_s = parser.train(gold_sent, i)
            correct_iter += correct_s
            all_iter += all_s

        print('fraction of correct transitions iteration %d: %d/%d = %f' % (i, correct_iter, all_iter, correct_iter/float(all_iter)))
    parser.avg_weights()
    test = filter_non_projective(fileio.read_conll_deps(opts.test))

    all_arcs = 0
    correct_arcs = 0

    for gold_test_sent in test:

        gold_arcs = set([(gold_test_sent[i][2], i) for i in range(len(gold_test_sent))])
        arcs = set(parser.run(gold_test_sent))
        correct_arcs += len(gold_arcs & arcs)
        all_arcs += len(gold_arcs)

    print('accuracy %d/%d = %f' % (correct_arcs, all_arcs, float(correct_arcs)/float(all_arcs)))
