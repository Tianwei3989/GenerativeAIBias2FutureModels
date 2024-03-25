# Corrupting Bias By GenerativeAI
This project is about the study of Dataset Bias Corruption from GenerativeAI, results in the paper [**_Would Deep Generative Models Amplify Bias in Future Models?_**](https://drive.google.com/file/d/1HOWYVf84zc-smpWeGuCWDbZzG3T78Tw6/view?usp=sharing)

<p align="center">
         <img src="https://github.com/Tianwei3989/CorruptingBiasByGenerativeAI/blob/main/imgs/Introduction.png" width="80%">
</p>

The goal of this repository is to back up all experiments code, as well as the training and evaluation manner, to github for further usage. 
This repository mainly contains codes for three parts: image generation (Stable Diffusion), pre-training model (OpenCLIP), and evaluation (PHASE, COCO bias, self similarity (SS), person preference (PP), and GeoDE).

## Installation
Basic setting
```
conda create -n sd_bias python=3.8.10
conda activate sd_bias
```

For extracting CC3M data, an additional package should be installed:
```
pip install img2dataset
```

### Stable Diffusion
```
pip install diffusers==0.10.2 transformers scipy ftfy accelerate
```

### OpenCLIP
```
pip install open_clip_torch[training]
```
We refered to the codes from [OpenCLIP](https://github.com/mlfoundations/open_clip).

## Data Preparation
### CC3M
Please download the CC3M captions from the [offical cite](https://ai.google.com/research/ConceptualCaptions/).


Following OpenCLIP, we extract CC3M by using [img2dataset](https://github.com/rom1504/img2dataset/tree/main), by
```
img2dataset --url_list Train_GCC-training.tsv --input_format "tsv"\
         --url_col "url" --caption_col "caption" --output_format webdataset\
           --output_folder cc3m --processes_count 16 --thread_count 64 --image_size 256\
             --enable_wandb True
```

### CC3M_SD
#### Captions
We manually combine the caption with its index by a character `#`, e.g., `0#a very typical bus station`.
The process could be done by 
```python
import numpy as np
import pandas as pd
df = pd.read_csv("./Train_GCC-training.tsv", header=None, sep='\t').iloc[:,0].reset_index()
df['prompt'] = df['index'].astype(str) + '#' + df.iloc[:,1]

prompts = df_['prompt'].values
np.savetxt(r"./prompts_train.txt", save_prompts, fmt='%s')

```

#### Image generation
Run `txt2img_cc3m.py` by 
```
python txt2img_cc3m.py --from_file ./prompts_train.txt --outdir ./SD_images --seed 123
```
**_CAUTION_**: The generated image would take more than 1.4TB of storage. Please prepare enough space in advance.

#### Compress to tar files
Run `create_mix_gcc.py` by
```
python create_mix_gcc.py --cc3m_path ./cc3m --sd_cc3m_path ./SD_images --outdir ./SD_images\
     --seed 123 --mix_ratio 0.2 --outdir ./cc3m_mix_
```
The parameter `--mix_radio` decides the ratio of the generated images to mix with the original CC3M.
Please set it to the number you want. 
If you run the above code, the output would be a folder `./cc3m_mix_020/`, i.e., a CC3M dataset mixing with 20% of generated images.

#### Our CC3M generated data
Our generated images are released [here](https://huggingface.co/datasets/T3989/SD_Bias_CC3M).
The images as well as the captions have already been compressed to tar file, literally, the ``cc3m_mix_100``.
You can extract them to prepare your mixing CC3M dataset by the above codes.


### Evaluation datasets

Please request the required dataset from their official website (links are attached to the dataset name).

[PHASE](https://github.com/noagarcia/phase)

[COCO bias](https://princetonvisualai.github.io/imagecaptioning-bias/)
> If you retrieve images from the original COCO dataset, you can use the following scripts for constructing the required data for the COCO bias dataset.
>
> ```python
> import shutil
> import os
> import pandas as pd
>
> data_root = <your COCO bias files download path>
> image_root = <your COCO images download path>
> 
> df_img = pd.read_csv(os.path.join(data_root, 'images_val2014.csv'))
> df_img_val = df_img[df_img['split']=='val']
> 
> dst_img_path = os.path.join(data_root, 'images/val/')
> os.makedirs(dst_img_path, exist_ok=True)
> df_bias = pd.DataFrame()
> for i in df_img_val.id.values:
>     image_name = 'COCO_val2014_' + str(i).zfill(12) + '.jpg'
>     shutil.copyfile(os.path.join(image_root, image_name), os.path.join(dst_img_path, image_name))
> 
> ```

[FairFace](https://github.com/joojs/fairface) (We use the version of ``Padding=1.25``)

[GeoDE](https://geodiverse-data-collection.cs.princeton.edu/)


## Train OpenCLIP from scratch
Please use the following command to train OpenCLIP with CC3M.
```
torchrun --nproc_per_node 4 -m training.main --epochs 50 --precision amp --workers 4\
         --train-num-samples 3318333 --dataset-type webdataset --batch-size 360\
                   --train-data './cc3m_mix_020/{00000..00331}.tar'\
                           --name cc3m_mix_020
```
If you want to train a model using CC3M with different radio of generated images, please prepare the corresponding dataset according to [Data Preparation](https://github.com/Tianwei3989/GenerativeAIBias2FutureModels?tab=readme-ov-file#data-preparation) and change ``--train-data './cc3m_mix_020/{00000..00331}.tar'`` to the directory you set, e.g., ``cc3m_mix_040``, ``cc3m_mix_060``, ``cc3m_mix_080``, etc.

When setting ``--name cc3m_mix_020``, you can find the trained model from ``./logs/cc3m_mix_020/checkpoints/``.

To simplify the file indexing and save storage space, we recommend you only save the final model ``epoch_50.pt`` to ``./logs/cc3m_mix_020/``, which may also ease the following bias evaluations.
A simple process could be running the command: ``mkdir -p ./models/cc3m_mix_020/ && mv ./logs/mix_cc3m_020/checkpoints/epoch_50.pt ./models/cc3m_mix_020/epoch_50.pt``.

## Bias evaluation

### PHASE image retrieval
Run ``./tools/image_retrieval.py`` to extract features from PHASE dataset by 
```
python ./tools/image_retrieval.py --data_path <your data root path> --task PHASE\
         --model RN50 --from_pretrain <model ratio code (e.g., 020), or pre-trained model name>\
                  --model_path <the path to all trained models>
```
The output features are saved in  ``data_path/Images/`` by default.

Then, run ``./bias_metrics/phase_evaluation.py`` by 
```
python ./bias_metrics/phase_evaluation.py --data_path <your data root path>\
         --feature_root <your feature root path> --model_name RN50\
                  --from_pretrain <model ratio code, or pre-trained model name>
```

### COCO bias image retrieval
Run ``./tools/image_retrieval.py`` by 
```
python ./tools/image_retrieval.py --data_path <your data root path> --task coco_bias\
         --model RN50 --from_pretrain <model ratio code, or pre-trained model name>\
                  --model_path <the path to all trained models>
```
The output features are saved in  ``data_path/Images/`` by default.

Then, run ``./bias_metrics/COCO_bias_evaluation.py`` by 
```
python ./bias_metrics/COCO_bias_evaluation.py --data_path <your data root path>\
         --feature_root <your feature root path> --model_name RN50\
                  --from_pretrain <model ratio code, or pre-trained model name>
```

### Self similarity
Run ``./tools/extract_feature.py`` by 
```
python ./tools/extract_feature.py --data_path <your data root path> --task FairFace\
         --model RN50 --from_pretrain <model ratio code, or pre-trained model name>\
                  --model_path <the path to all trained models>
```
The output features are saved in  ``data_path/`` by default.

Then, run ``./bias_metrics/markedness_self_similarity.py`` by 
```
python ./bias_metrics/markedness_self_similarity.py --data_path <your data root path>\
         --model RN50 --from_pretrain <model ratio code, or pre-trained model name>
```
We refered to the codes from [Markedness](https://github.com/wolferobert3/visual_semantic_markedness).


### Person preference
Run ``./bias_metrics/markedness_person_preference.py`` by 
```
python ./bias_metrics/markedness_person_preference.py --data_path <your data root path>\
         --model_path <your data root path, only when use manually trained models>\
                  --model RN50 --output_root <path to save outputs>\
                            --from_pretrain <model ratio code, or pre-trained model name>
```
We refered to the codes from [Markedness](https://github.com/wolferobert3/visual_semantic_markedness).

## Results

Please find them [here](https://docs.google.com/spreadsheets/d/1rW8veKoOCI3f1C5baDFZmBhL03Re5N1vTvng7ZpbU3Q/edit?usp=sharing)
