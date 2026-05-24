from collections import Counter


class BPETokenizer:
    """Byte Pair Encoding entrenado sobre un texto.

    Vocabulario inicial: caracteres unicos del texto. Durante el
    entrenamiento se buscan los pares adyacentes mas frecuentes y se
    fusionan en nuevos tokens, hasta alcanzar `vocab_size` tokens.

    NOTA: para ser BPE de verdad, tendríamos que hacerlo sobre bytes, no sobre
    caracteres, pero para la práctica funciona bien.
    """

    def __init__(self, text, vocab_size=300):
        self.vocab_size = vocab_size
        # Inicializamos con caracteres encontrados en el texto
        self.vocab = sorted(set(text))  # vocab[id] -> token string.
        self.tok2id = {tok: i for i, tok in enumerate(self.vocab)}

        tokens = [self.tok2id[c] for c in text]
        self.merges = []  # lista de ((id_a, id_b), nuevo_id), para encode()

        # for new_id in range(len(self.vocab), vocab_size):
        #    pairs = Counter(zip(tokens, tokens[1:]))
        #    best = pairs.most_common(1)[0][0]

        stats = Counter(zip(tokens, tokens[1:]))
        for new_id in range(len(self.vocab), vocab_size):
            best = max(stats, key=stats.get)
            del stats[best]

            new_tok = self.vocab[best[0]] + self.vocab[best[1]]
            self.tok2id[new_tok] = new_id
            self.vocab.append(new_tok)
            self.merges.append((best, new_id))

            tokens = self._apply_merge(tokens, best[0], best[1], new_id)

    @staticmethod
    def _apply_merge(tokens, a, b, new_id, stats=None):
        nw_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                if stats is not None:
                    # Contexto izquierdo (basado en el último token ya procesado)
                    if len(nw_tokens) > 0:
                        prev_tok = nw_tokens[-1]
                        pair_left = (prev_tok, a)
                        stats[pair_left] -= 1
                        if stats[pair_left] <= 0:
                            del stats[pair_left]
                        stats[(prev_tok, new_id)] += 1

                    # Contexto derecho (basado en el próximo token sin procesar)
                    if i < len(tokens) - 2:
                        next_tok = tokens[i + 2]
                        pair_right = (b, next_tok)
                        stats[pair_right] -= 1
                        if stats[pair_right] <= 0:
                            del stats[pair_right]
                        stats[(new_id, next_tok)] += 1

                nw_tokens.append(new_id)
                i += 2
            else:
                nw_tokens.append(tokens[i])
                i += 1

        return nw_tokens

    def encode(self, text):
        """Codifica un texto aplicando los merges aprendidos."""
        tokens = [self.tok2id.get(c, 0) for c in text]
        for (a, b), new_id in self.merges:
            tokens = self._apply_merge(tokens, a, b, new_id)
        return tokens

    def decode(self, ids):
        """Decodifica una lista de ids a texto."""
        # Une las cadenas correspondientes a cada ID almacenadas en self.vocab
        return "".join(self.vocab[idx] for idx in ids)

    def __repr__(self):
        pretty = [t.replace("\n", "\\n").replace(" ", "▁") for t in self.vocab]
        return f"{len(self.vocab)} tokens: ['{"', '".join(pretty)}']"


# Si ejecutamos este módulo directamente, probamos el tokenizador
if __name__ == "__main__":
    import sys
    from pathlib import Path

    files_path = Path(sys.argv[1] if len(sys.argv) > 1 else "resources")
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    textos = "\n\n".join(open(p).read() for p in files_path.glob("*.txt"))
    tokenizer = BPETokenizer(textos, vocab_size=vocab_size)
    print(tokenizer)
