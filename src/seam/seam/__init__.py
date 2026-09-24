from .config import load_config

__all__ = ["SEAMModel", "load_config"]


def __getattr__(name):
    if name == "SEAMModel":
        from .modeling.model import SEAMModel

        return SEAMModel
    raise AttributeError(name)
