import numpy as np

from final_project import utils


class VectorsBase:
    name = None

    def __init__(self, *, token_to_id, special_tokens=None):
        self.dims = len(self._get_vector(next(iter(token_to_id.keys()))))
        self.token_to_id = token_to_id
        self.special_tokens = special_tokens

    def get_vector(self, token):
        if self.special_tokens is not None and token in self.special_tokens:
            return np.random.normal(0, 1, (self.dims,))
        return self._get_vector(token)

    def _get_vector(self, token):
        raise NotImplementedError()

    def get_vectors(self):
        max_id = max(self.token_to_id.values())
        vectors = np.random.normal(0, 1, (max_id + 1, self.dims))
        for token, token_id in self.token_to_id.items():
            vector = self._get_vector(token)
            if vector is not None and any(vector):
                vectors[token_id, :] = vector
        return vectors


class SpacyVectors(VectorsBase):
    def __init__(self, **kwargs):
        import spacy

        self.spacy_model = spacy.load(self.name)
        super().__init__(**kwargs)

    def _get_vector(self, token):
        return self.spacy_model.tokenizer(token).vector


class LargeEnglishSpacyVectors(SpacyVectors):
    name = "en_core_web_lg"


class MediumEnglishSpacyVectors(SpacyVectors):
    name = "en_core_web_md"


def get_vectors(config, tokenizer):
    if config.vectors is None:
        return None
    for sub in utils.get_subclasses(VectorsBase):
        if sub.name == config.vectors:
            return sub(
                token_to_id=tokenizer.token_to_id,
                special_tokens=tokenizer.special_tokens,
            ).get_vectors()
