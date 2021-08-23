from collections import Counter
from itertools import chain

from universal_transformer.class_registry import register_class, registry


class TokenizerBase:
    def __init__(
        self,
        *,
        lower,
        sos_token="[START]",
        eos_token="[EOS]",
        unknown_token="[UNK]",
        pad_token="[PAD]",
        max_vocab_size=float("inf"),
        **kwargs,
    ):
        self.id_to_token = None
        self.token_to_id = None
        self.vectors = None
        self.lower = lower
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unknown_token = unknown_token
        self.pad_token = pad_token
        self.special_tokens = (
            self.pad_token,
            self.eos_token,
            self.unknown_token,
            self.pad_token,
            self.sos_token,
        )
        self.special_tokens = tuple(
            token
            for token in (
                self.pad_token,
                self.eos_token,
                self.unknown_token,
                self.sos_token,
            )
            if token is not None
        )
        self.max_vocab_size = max_vocab_size

    def fit(self, *, texts=None, text_batch_iter=None):

        counter = Counter()

        def fit_batch(batch):
            if self.lower:
                batch = list(map(str.lower, batch))
            counter.update(list(chain(*self.tokenize(batch))))

        if texts is not None:
            fit_batch(texts)
        elif text_batch_iter is not None:
            for batch in text_batch_iter:
                fit_batch(batch)

        self.token_to_id = {}

        token_i = 0
        for token in self.special_tokens:
            self.token_to_id[token] = token_i
            token_i += 1

        # Not sure this is the place to enforce this, but it
        # does seem to be a convention.
        if self.pad_token is not None:
            assert self.token_to_id[self.pad_token] == 0

        for token, count in counter.most_common(min(self.max_vocab_size, len(counter))):
            self.token_to_id[token] = token_i
            token_i += 1

        self.id_to_token = {
            token_id: token for token, token_id in self.token_to_id.items()
        }

    def tokenize(self, texts):
        if self.lower:
            texts = [text.lower() for text in texts]
        return self._tokenize(texts)

    def _tokenize(self, texts):
        raise NotImplementedError()

    def encode(self, texts):
        if isinstance(texts, str):
            return self.encode([texts])[0]

        id_seqs = []
        for tokens in self.tokenize(texts):
            id_seq = []
            for token in tokens:
                if token not in self.token_to_id:
                    token = self.unknown_token
                id_seq.append(self.token_to_id[token])
            id_seqs.append(id_seq)

        return self._post_process(
            id_seqs,
            pad_id=self.token_to_id[self.pad_token] if self.pad_token else None,
            sos_id=self.token_to_id[self.sos_token] if self.sos_token else None,
            eos_id=self.token_to_id[self.eos_token] if self.eos_token else None,
        )

    @staticmethod
    def _post_process(
        ids_seqs, *, pad_id=None, sos_id=None, eos_id=None, max_length=float("inf")
    ):
        extra_tokens = 0
        if sos_id is not None:
            extra_tokens += 1
        if eos_id is not None:
            extra_tokens += 1

        new_id_seqs = []
        for id_seq in ids_seqs:
            new_seq = []
            if sos_id is not None:
                new_seq.append(sos_id)
            if len(id_seq) + extra_tokens > max_length:
                new_seq.extend(id_seq[: max_length - extra_tokens])
            else:
                new_seq.extend(id_seq)
            if eos_id is not None:
                new_seq.append(eos_id)
            new_id_seqs.append(new_seq)

        longest_seq_len = max(map(len, new_id_seqs))
        for id_seq in new_id_seqs:
            for _ in range(longest_seq_len - len(id_seq)):
                id_seq.append(pad_id)

        return new_id_seqs

    def decode(self, id_seqs):
        output = []
        for id_seq in id_seqs:
            decoded = " ".join(self.id_to_token[id] for id in id_seq)
            output.append(decoded)
        return output


@register_class(("tokenizer", "en_core_web_md"), name="en_core_web_md", lower=True)
@register_class(("tokenizer", "en_core_web_lg"), name="en_core_web_lg", lower=True)
@register_class(("tokenizer", "de_core_news_md"), name="de_core_news_md", lower=True)
class SpacyTokenizer(TokenizerBase):
    def __init__(self, name, **kwargs):
        super().__init__(hash_key=name, **kwargs)
        import spacy

        self.spacy_model = spacy.load(
            name, disable=["ner", "tagger", "parser", "attribute_ruler", "lemmatizer"]
        )

    def _tokenize(self, texts):
        docs = self.spacy_model.pipe(
            texts,
            disable=["ner", "tagger", "parser", "attribute_ruler", "lemmatizer"],
        )
        tokens = []
        for doc in docs:
            tokens.append(tuple(token.text for token in doc))
        return tokens


@register_class(
    ("tokenizer", "hugging_face_word_level"), lower=True, max_vocab_size=20000
)
class HuggingFaceWordLevelTokenizer(TokenizerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

        self.tokenizer = Tokenizer(models.WordLevel(unk_token=self.unknown_token))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        if self.lower:
            self.tokenizer.normalizer = normalizers.Lowercase()

    def fit(self, *, texts=None, text_batch_iter=None, max_tokens=None):
        from tokenizers import trainers

        trainer = trainers.WordLevelTrainer(
            vocab_size=self.max_vocab_size, special_tokens=list(self.special_tokens)
        )
        self.tokenizer.train_from_iterator(text_batch_iter, trainer=trainer)
        self.token_to_id = self.tokenizer.get_vocab()
        self.id_to_token = {
            token_id: token for token, token_id in self.token_to_id.items()
        }

    def encode(self, texts):
        id_seqs = self.tokenizer.encode_batch(texts)
        id_seqs = [id_seq.ids for id_seq in id_seqs]
        return self._post_process(
            id_seqs,
            pad_id=self.token_to_id[self.pad_token] if self.pad_token else None,
            sos_id=self.token_to_id[self.sos_token] if self.sos_token else None,
            eos_id=self.token_to_id[self.eos_token] if self.eos_token else None,
        )

    def decode(self, id_seqs):
        self.tokenizer.decode_batch(id_seqs)


def get_tokenizers(config):
    # TODO: Had to hack this up pretty good to handle two tokenizers.

    def create_tokenizer(tokenizer_name):
        cls, kwargs = registry[("tokenizer", tokenizer_name)]
        accepted_args = set(cls.__init__.__code__.co_varnames)
        accepted_args.remove("self")
        kwargs.update(
            {
                k.replace("tokenizer.", ""): v
                for k, v in config.items()
                if "tokenizer." in k
            }
        )
        return cls(**kwargs)

    tokenizer = None
    if config.tokenizer is not None:
        tokenizer = create_tokenizer(config.tokenizer)
    output_tokenizer = None
    if config.output_tokenizer is not None:
        output_tokenizer = create_tokenizer(config.output_tokenizer)

    return tokenizer, output_tokenizer
