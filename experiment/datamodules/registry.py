_datamodules = {}


def register_datamodule(name):
    def decorator(cls):
        _datamodules[name] = cls
        return cls

    return decorator


def build_datamodule(**kwargs):
    assert "conf" in kwargs
    conf = kwargs["conf"]
    key = f"{conf.data.name}_kasm"
    if key not in _datamodules:
        raise KeyError(f"Unknown KASM datamodule: {key}")
    return _datamodules[key](**kwargs)
