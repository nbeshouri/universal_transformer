class ConfigWrapper:
    def __init__(self, config):
        self.config = config

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getattr__(self, key):
        try:
            return getattr(self.config, key)
        except:
            pass

    def __setattr__(self, key, value):
        if key == "config":
            self.__dict__["config"] = value
        else:
            setattr(self.config, key, value)
