# mwmae-jax-official
This is the official repository for the paper ["Masked Autoencoders with Multi-Window Local-Global Attention Are Better Audio Learners"](https://arxiv.org/abs/2306.00561), to appear at the Twelfth International Conference on Learning Representations (ICLR) 2024.


# Contents
* [Pre-trained weights for the default configurations](https://drive.google.com/drive/folders/1tM723MLFJRaWVmLVjE49pJ0Yi3lpMZHV?usp=sharing)
* Our local copy of [hear-eval-kit](external_sources/hear-eval-kit) for easy downstream reproducibility. Original can be found [here](https://github.com/hearbenchmark/hear-eval-kit)
* [Feature extraction API](hear_api) compatible with the [hear-eval-kit](https://github.com/hearbenchmark/hear-eval-kit) format for extracting features.
* Helper code to [extract features](extract_features.sh) and [run downstream experiments](downstream_experiments.sh) on provided pre-trained models

---

# Setup

## Environment
* Required: cuda 11.x, cudnn 8.2 or newer.
* create a new conda environment with python 3.9
* Setting up `jax==0.4.18`. We used jax because we wanted a framework that worked seemlessly across GPUs and TPUs, and since most of the experiments were done on a TPU.
* You need `torch` to run downstream experiments (hear-eval-kit is based on torch)

Follow these steps
```shell
conda create -n mwmae-env python=3.9 -y
conda activate mwmae-env

# install pre jax requirments
pip install -r pre_jax_requirements.txt

# install hear-eval-kit specific requirements
pip install -r hear-eval-kit/requirements.txt

# install hear-eval-kit, WITHOUT AUTO DEPS
cd external_sources/hear-eval-kit && pip install --no-deps . && cd -

# install jax
pip install --upgrade "jax[cuda]==0.4.18" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install flax and other jax requirements
pip install -r post_jax_requirements.txt

```

## Get 16000 Hz data from hear
* Follow https://hearbenchmark.com/hear-tasks.html to get data. By default, data on HEAR's zenodo page is 48000 Hz.
* We recommend downloading data directly from HEAR's [GCS bucket](gs://hear2021-archive/tasks/), where you can find preprocessed 16000 Hz data.
* Extract all the files to a folder `$TASKS_DIR`

## Get pretrained weights

* Pre-trained can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1tM723MLFJRaWVmLVjE49pJ0Yi3lpMZHV?usp=sharing)
* Download the entire folder and export that folder as `$MW_MAE_MODEL_DIR`

## Extract features

```shell
export MW_MAE_MODEL_DIR=/path/to/pretrained_weights
./extract_features.sh $TASKS_DIR $OUTPUT_DIR
```
where TASKS_DIR is the directory where you extracted tasks from HEAR-2021 to, and OUTPUT_DIR is the base directory where output features will be stored.
This also prepares a `todo_audioset` directory in OUTPUT_DIR, which is setting up for downstream classification on 10 seeds.

## Run downstream experiments

```shell
./downstream_experiments.sh mw_mae_200_16x4_384d_8h_4l $OUTPUT_DIR
```

This will run downstream experiments on all the extracted features on 10 random seeds.

## Get results
Finally, you can run the following script to get results of downstream experiments of the two models

```shell
python stats_aggregation.py --base_dir ${OUTPUT_DIR}/todo_audioset --output_dir ${OUTPUT_DIR}/parsed_results
```

---

# Extracting features on your own audio file
The [hear_api](hear_api) can be used to extract features from your own audio files.

```python
import torchaudio


from hear_api import RuntimeMAE
from importlib import import_module
config = import_module("configs.pretraining.mwmae_base_200_4x16_precomputed").get_config()
mae = RuntimeMAE(config, "path/to/pretrained/weights")

# alternatively just use the following if you have the paths setup right
# mae = import_module("hear_api.mwmae_base_200_4x16_384d-8h-4l").load_model()

x, sr = torchaudio.load("path/to/audio.wav")
o = mae.get_scene_embeddings(x)

```

Even though the API is a mix of pytorch and jax code, inference works quite fast. You can use the code for inspiration to write your own native jax/flax inference API.

---

# Pretraining
Pretraining code is included in the release, but since we cannot redistribute AudioSet we're working on a FSD50K example.  
Either way, the simple command to train the `mwmae_base_200_4x16_precomputed` model is with the command:
```
python main.py --config configs/pretraining/mwmae_base_200_4x16_precomputed.py --workdir $PWD/mwmae_base_200_4x16_8x128_default_bfloat16 --no_wandb
```

---