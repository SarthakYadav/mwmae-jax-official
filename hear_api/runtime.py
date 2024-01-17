import jax
import numpy as np
import jax.numpy as jnp
import sys
sys.path.append('..')
import torch
from .feature_helper import LogMelSpec, get_timestamps
from src.trainer import MAETrainer
from functools import partial


def get_grid_size(img_size, patch_size):
    grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
    return grid_size


def forward(batch, state, model):
    variables = {
        'params': state.get_all_params,                    # absolutely ok to just use state.get_all_params here
        'batch_stats': state.batch_stats,
        "buffers": state.buffers
    }
    logits = model.apply(
        variables, batch, train=False, mutable=False, method=model.forward_features
    )
    return logits


class RuntimeMAE(torch.nn.Module):
    def __init__(self, config, weights_dir) -> None:
        super().__init__()
        self.config = config
        self.mae_trainer = MAETrainer(config, weights_dir, True, seed=0, inference=True)
        self.forward_jit = jax.jit(partial(forward, state=self.mae_trainer.state, model=self.mae_trainer.model))
        self.log_mel_spec = LogMelSpec()
        self.grid_size = get_grid_size(img_size=self.mae_trainer.model.img_size, patch_size=self.mae_trainer.model.patch_size)
        self.input_size = self.mae_trainer.model.img_size
        self.embed_dim = self.mae_trainer.model.embed_dim
        self.sample_rate = 16000
    
    def to_feature(self, batch_audio):
        x = self.log_mel_spec(batch_audio)
        mean = torch.mean(x, [1, 2], keepdims=True)
        std = torch.std(x, [1, 2], keepdims=True)

        x = (x - mean) / (std + 1e-8)

        x = x.permute(0, 2, 1)
        x = jnp.asarray(x.detach().cpu().numpy())
        x = x[Ellipsis, jnp.newaxis]
        return x
    
    def encode(self, lms):
        x = lms

        patch_fbins = self.grid_size[1]
        unit_frames = self.input_size[0]

        embed_d = self.embed_dim

        cur_frames = x.shape[1]
        pad_frames = unit_frames - (cur_frames % unit_frames)
        if pad_frames > 0:
            pad_arg = [(0, 0), (0, pad_frames), (0, 0), (0, 0)]
            x = jnp.pad(x, pad_arg, mode="reflect")

        embeddings = []
        for i in range(x.shape[1] // unit_frames):
            x_inp = x[:, i*unit_frames:(i+1)*unit_frames, Ellipsis]
            logits = self.forward_jit(x_inp)
            embeddings.append(logits)
        x = jnp.concatenate(embeddings, axis=1)
        pad_emb_frames = int(embeddings[0].shape[1] * pad_frames / unit_frames)
        if pad_emb_frames > 0:
            x = x[:, :-pad_emb_frames, Ellipsis]
        x = x.astype(jnp.float32)
        return x
    
    def audio2feats(self, audio):
        x = self.to_feature(audio)
        x = self.encode(x)
        x = torch.from_numpy(np.array(x.copy()))
        return x
    
    def get_scene_embeddings(self, audio):
        x = self.audio2feats(audio)        
        x = torch.mean(x, dim=1)
        return x
    
    def get_timestamp_embeddings(self, audio):
        x = self.audio2feats(audio)
        ts = get_timestamps(self.sample_rate, audio, x)
        return x, ts
    