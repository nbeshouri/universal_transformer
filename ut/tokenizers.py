from collections import Counter
from itertools import chain

from transformers import *

from ut import utils


class TokenizerBase:
    name = None

    def __init__(
        self,
        *,
        lower,
        sos_token="[START]",
        eos_token="[EOS]",
        unknown_token="[UNK]",
        pad_token="[PAD]",
        pad=True,
        seq_length_max=500,
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
        self.pad = pad
        self.seq_length_max = seq_length_max
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

    def fit(self, texts, max_tokens=None):
        if self.lower:
            texts = map(str.lower, texts)
        tokens = tuple(chain(*map(self._tokenize, texts)))
        if max_tokens is None:
            max_tokens = len(set(tokens))

        self.token_to_id = {}

        token_i = 0
        for token in self.special_tokens:
            self.token_to_id[token] = token_i
            token_i += 1

        # Not sure this is the place to enforce this, but it
        # does seem to be a convention.
        if self.pad_token is not None:
            assert self.token_to_id[self.pad_token] == 0

        for token, count in Counter(tokens).most_common(max_tokens):
            self.token_to_id[token] = token_i
            token_i += 1

        self.id_to_token = {
            token_id: token for token, token_id in self.token_to_id.items()
        }

    def tokenize(self, text):
        if self.lower:
            text = text.lower()

        tokens = []

        if self.sos_token is not None:
            tokens.append(self.sos_token)

        for token in self._tokenize(text):
            if self.token_to_id is not None and token not in self.token_to_id:
                token = self.unknown_token
            tokens.append(token)

        if self.eos_token is not None:
            tokens.append(self.eos_token)

        return tokens

    def _tokenize(self, text):
        raise NotImplementedError()

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.token_to_id[token] for token in tokens]

    def encode_texts(self, texts):
        encoded_texts = list(map(self.encode, texts))
        max_len = max(map(len, encoded_texts))
        for encoded_text in encoded_texts:
            for _ in range(max_len - len(encoded_text)):
                encoded_text.append(self.token_to_id[self.pad_token])
        return encoded_texts

    def decode(self, ids):
        " ".join(self.id_to_token[id] for id in ids)


class SpacyTokenizer(TokenizerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import spacy

        self.spacy_model = spacy.load(self.name)

    def _tokenize(self, text):
        return tuple(token.text for token in self.spacy_model.tokenizer(text))


class LargeEnglishSpacyTokenizer(SpacyTokenizer):
    name = "en_core_web_lg"

    def __init__(self):
        super().__init__(lower=True)


class MediumEnglishSpacyTokenizer(SpacyTokenizer):
    name = "en_core_web_md"

    def __init__(self):
        super().__init__(lower=True)


class HuggingFaceTokenizer(TokenizerBase):
    def __init__(self, hugging_face_tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.hugging_face_tokenizer = hugging_face_tokenizer

    def fit(self, texts, max_tokens=None):
        pass

    def encode(self, text):
        ids = self.hugging_face_tokenizer.encode(
            text, max_length=self.seq_length_max, truncation=True
        )
        for _ in range(self.seq_length_max - len(ids)):
            ids.append(0)
        return ids

    def decode(self, ids):
        return self.hugging_face_tokenizer.decode(ids)


def get_tokenizer(config):
    for sub in utils.get_subclasses(TokenizerBase):
        if sub.name == config.tokenizer:
            # If necessary, could feed the relevant config stuff here.
            # The basic idea is that the "name" of the tokenizer should
            # be a class thing even if there might be config options
            # that otherwise modify things.
            return sub()

    try:
        hugging_face_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        HuggingFaceTokenizer(hugging_face_tokenizer, config.tokenizer)
    except OSError:
        pass

    raise ValueError("Tokenizer not found!")


# TODO for next time: You need to make that subclass search thing
#  recursive. Also, like, you need to
