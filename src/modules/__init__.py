from . import mae
from ..helpers.utilities import get_dtype
from ml_collections import ConfigDict


def create_model(config: ConfigDict, precision="float32"):
    model_dtype = get_dtype(precision)
    model_cls = getattr(mae, config.model.arch)
    model_args = config.model.get("model_args", {})
    # model_args["dtype"] = model_dtype
    model = model_cls(**model_args, dtype=model_dtype)
    return model
