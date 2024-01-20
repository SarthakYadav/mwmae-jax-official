import os
import sys
from hear_api.runtime import RuntimeMAE
import ml_collections
from importlib import import_module
MW_MAE_MODEL_DIR = os.environ.get("MW_MAE_MODEL_DIR")


config_path = "configs.pretraining.mae_base_200_4x16_precomputed"
RUN_ID = 1
model_path = os.path.join(MW_MAE_MODEL_DIR, f"mae_base_200_4x16_8x128_default_run{RUN_ID}")


def load_model(model_path=model_path, config=import_module(config_path).get_config()):
    model = RuntimeMAE(config, model_path)
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
