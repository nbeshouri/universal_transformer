from collections import UserDict


class ClassRegistry(UserDict):
    def __getitem__(self, key):
        cls, kwargs = super().__getitem__(key)
        return cls, kwargs.copy()

    def __setitem__(self, key, value):
        raise KeyError("Use register_class!")

    def register_class(self, key, **kwargs):
        def _register_class(cls):
            super(ClassRegistry, self).__setitem__(key, (cls, kwargs))
            return cls

        return _register_class


registry = ClassRegistry()
register_class = registry.register_class
