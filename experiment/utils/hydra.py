from omegaconf import DictConfig


def fix_relative_path(conf):
    for key in conf:
        value = conf[key]
        if isinstance(value, str) and key.endswith("dir"):
            if not value.startswith(conf.root_dir):
                conf[key] = f"{conf.root_dir}/{value}"
        elif isinstance(value, DictConfig):
            fix_relative_path(value)
