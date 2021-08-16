import numpy as np

from universal_transformer.class_registry import register_class, registry


class VectorsBase:
    def __init__(self, *, token_to_id, special_tokens=None, vector_size=None):
        if vector_size is None:
            self.vector_size = len(self.get_vector(next(iter(token_to_id.keys()))))
        else:
            self.vector_size = vector_size
        self.token_to_id = token_to_id
        self.special_tokens = special_tokens

    def get_vector(self, token):
        raise NotImplementedError()

    def get_vectors(self):
        max_id = max(self.token_to_id.values())
        vectors = np.random.normal(0, 1, (max_id + 1, self.vector_size))
        for token, token_id in self.token_to_id.items():
            vector = self.get_vector(token)
            if vector is not None and any(vector):
                vectors[token_id, :] = vector
        return vectors


@register_class(("vectors", "random"))
class RandomVectors(VectorsBase):
    def get_vector(self, token):
        return None


@register_class(("vectors", "en_core_web_md"), name="en_core_web_md")
@register_class(("vectors", "en_core_web_lg"), name="en_core_web_lg")
@register_class(("vectors", "de_core_news_md"), name="de_core_news_md")
class SpacyVectors(VectorsBase):
    def __init__(self, name, **kwargs):
        import spacy

        self.spacy_model = spacy.load(name)
        super().__init__(**kwargs)

    def get_vector(self, token):
        vector = self.spacy_model.tokenizer(token).vector
        # If spacy doesn't know the word, it maps to zero vector.
        # I'd rather use a
        if not vector.any():
            vector = np.random.normal(0, 1, vector.shape)
        return vector


def get_vectors(config, tokenizer):
    if config.vectors is None:
        return None
    key = ("vectors", config.vectors)
    if key in registry:
        cls, kwargs = registry[key]
        accepted_args = set(cls.__init__.__code__.co_varnames)
        accepted_args.remove("self")
        kwargs.update(
            {k.replace("vectors.", ""): v for k, v in config.items() if "vectors." in k}
        )
        vectors_obj = cls(
            token_to_id=tokenizer.token_to_id,
            special_tokens=tokenizer.special_tokens,
            **kwargs
        )
        return vectors_obj.get_vectors()
    raise KeyError("Vectors not found!")
