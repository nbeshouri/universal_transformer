import numpy as np

from universal_transformer.class_registry import registry, register_class


class VectorsBase:
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


@register_class(("vectors", "en_core_web_md"), name="en_core_web_md")
@register_class(("vectors", "en_core_web_lg"), name="en_core_web_lg")
class SpacyVectors(VectorsBase):
    def __init__(self, name, **kwargs):
        import spacy

        self.spacy_model = spacy.load(name)
        super().__init__(**kwargs)

    def _get_vector(self, token):
        return self.spacy_model.tokenizer(token).vector


def get_vectors(config, tokenizer):
    if config.vectors is None:
        return None
    key = ("vectors", config.vectors)
    if key in registry:
        cls, kwargs = registry[key]
        vectors_obj = cls(
            token_to_id=tokenizer.token_to_id,
            special_tokens=tokenizer.special_tokens,
            **kwargs
        )
        return vectors_obj.get_vectors()
    raise KeyError("Vectors not found!")
