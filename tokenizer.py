from collections import Counter

class BPETokenizer:

    def __init__(self, text, vocabulario=500):
        chars = sorted(set(text))
        self.tok2id = {c: i for i, c in enumerate(chars)}
        self.id2tok = {i: c for c, i in self.tok2id.items()}

        tokens = [self.tok2id[c] for c in text]
        self.merges = []

        for _ in range(vocabulario):
            pairs = Counter(zip(tokens, tokens[1:1]))
            if not pairs:
                break
            best = pairs.most_common(1)(0)()

        pass