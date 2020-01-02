from collections import defaultdict

class Classifier:

    def __init__(self, labels):
        pass
    
    def score(self, fv):
        """
        Score a feature vector
        
        :param fv: A feature vector
        """
        pass



    def create_trainer(self):
        pass


class Trainer:
    def __init__(self, model):
        self.model = model

    def update(self, truth, guess, features):
        """
        Update the model (during training)
        
        :param truth: The true label
        :param guess: The guessed label
        :param features: The features to update
        """

    def finish(self):
        pass


    
class AveragedPerceptronTrainer(Trainer):
    def __init__(self, model):
        super(AveragedPerceptronTrainer, self).__init__(model)
        self.train_tick = 0
        self.train_last_tick = defaultdict(lambda: 0)
        self.train_totals = defaultdict(lambda: 0)

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
        # feature is a dictionary
        for f in features.keys():
            update_feature_label(truth, f, 1.0)
            update_feature_label(guess, f, -1.0)

    def finish(self):
        for fj in self.model.weights:
            for label in self.model.weights[fj]:
                total = self.train_totals[(fj, label)]
                t_delt = self.train_tick - self.train_last_tick[(fj, label)]
                total += t_delt * self.model.weights[fj][label]
                avg = round(total / float(self.train_tick))
                if avg:
                    self.model.weights[fj][label] = avg

        
class AveragedPerceptronClassifier(Classifier):
    def __init__(self, labels):
        super(AveragedPerceptronClassifier, self).__init__(labels)
        self.weights = {}
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

    def create_trainer(self):
        return AveragedPerceptronTrainer(self)



def init_embeddings(vocab_size, embed_dim, unif):
    return np.random.uniform(-unif, unif, (vocab_size, embed_dim))
    

class EmbeddingsReader:

    @staticmethod
    def from_text(filename, vocab, unif=0.25):
        
        with io.open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n ")
                values = line.split(" ")

                if i == 0:
                    # fastText style
                    if len(values) == 2:
                        weight = init_embeddings(len(vocab), values[1], unif)
                        continue
                    # glove style
                    else:
                        weight = init_embeddings(len(vocab), len(values[1:]), unif)
                word = values[0]
                if word in vocab:
                    vec = np.asarray(values[1:], dtype=np.float32)
                    weight[vocab[word]] = vec
        if '[PAD]' in vocab:
            weight[vocab['[PAD]']] = 0.0
        
        embeddings = nn.Embedding(weight.shape[0], weight.shape[1])
        embeddings.weight = nn.Parameter(torch.from_numpy(weight).float())
        return embeddings, weight.shape[1]
    
    @staticmethod
    def from_binary(filename, vocab, unif=0.25):
        def read_word(f):

            s = bytearray()
            ch = f.read(1)

            while ch != b' ':
                s.extend(ch)
                ch = f.read(1)
            s = s.decode('utf-8')
            # Only strip out normal space and \n not other spaces which are words.
            return s.strip(' \n')

        vocab_size = len(vocab)
        with io.open(filename, "rb") as f:
            header = f.readline()
            file_vocab_size, embed_dim = map(int, header.split())
            weight = init_embeddings(len(vocab), embed_dim, unif)
            if '[PAD]' in vocab:
                weight[vocab['[PAD]']] = 0.0
            width = 4 * embed_dim
            for i in range(file_vocab_size):
                word = read_word(f)
                raw = f.read(width)
                if word in vocab:
                    vec = np.fromstring(raw, dtype=np.float32)
                    weight[vocab[word]] = vec
        embeddings = nn.Embedding(weight.shape[0], weight.shape[1])
        embeddings.weight = nn.Parameter(torch.from_numpy(weight).float())
        return embeddings, embed_dim

class DNNClassifier(Classifier):
    def __init__(self, labels, embeddings_file, num_hidden=100):
        embeddings, dsz = EmbeddingsReader.from_text(embeddings_file)
        self._model = nn.Sequential(embeddings,
                                    nn.Linear(dsz, num_hidden),
                                    nn.Linear(num_hidden, labels),
                                    nn.LogSoftmax(dim=-1))
    def score():
        scored = self.model(fv)
        return scored
    
    
class DNNTrainer(Trainer):

    def __init__(self, model, batchsz=10, lr=0.001, eps=1e-8, beta1=0.9, beta2=0.999, wd=0, global_step=0, **kwargs):
        self.model = model
        self.global_step = global_step
        self.batchsz = batchsz
        self.crit = NLLLoss()
        self.optimizer = torch.optim.Adam(
            model._model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=wd
        )
        self.optimizer.zero_grad()
        
    def update(self, truth, guess, features):
        self.global_step += 1
        pred = self.model._model(features)
        loss = self.crit(pred, y)
        report_loss = loss.item()
        loss.backward()
        if self.global_step % self.batchsz == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
