import torch
from PIL import Image
import open_clip
import random
import numpy as np
import pandas as pd
import os
import argparse
import copy
from itertools import islice
from tqdm import tqdm, trange
import pickle

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed", type=int, default=0, help="Default random seed."
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--task",
    type=str,
    default='FairFace',
    help="in [FairFace, geode]",
)
parser.add_argument(
    "--data_path",
    type=str,
    help="the address to the evaluation datasets",
)
parser.add_argument(
    "--text_file",
    default=None,
    help="The name of text file. If None, skip the process the texts",
)
parser.add_argument(
    "--image_file",
    default='margin125/train',
    help="The name of image folder. If None, skip the process the images, candidates: [margin125/train, phase_person]",
)
parser.add_argument(
    "--model",
    type=str,
    default='RN50',
    help="model type in [RN50, ViT-B-32]",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="the address to the trained models",
)
parser.add_argument(
    "--from_pretrain",
    type=str,
    default='020',
    help="openai, cc12m, laion400m_e32, or an address to a pretrained model id (000, 020, etc.)",
)
parser.add_argument(
    "--epoch",
    type=int,
    default=50,
    help="The number of epoches that the model are trained, default 50 (i.e., epoch_XX.pt)",
)
args = parser.parse_args()

print('Extracting feature for TASK:', args.task)

random_seed(args.seed, 0)

if args.from_pretrain in ['openai','cc12m', 'laion400m_e32']:
    task_name = args.from_pretrain
    model_path = args.from_pretrain
else:
    task_name = 'cc3m_mix_' + args.from_pretrain # 'cc3m'
    model_path = os.path.join(args.model_path + 'cc3m_mix_' + args.from_pretrain, 'epoch_'+str(args.epoch)+'.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, _, preprocess = open_clip.create_model_and_transforms(
    args.model,
    pretrained=model_path,
    device=device,
)
tokenizer = open_clip.get_tokenizer(args.model)


# extract feature from prompts and images
if args.text_file is not None:
    file_name = os.path.join(args.data_path, args.task, args.text_file)
    print(f"reading prompts from {file_name}")

    if file_name[-3:] == 'txt':
        with open(file_name, "r") as f:
            prompts = f.read().splitlines()
        prompt_ids = range(len(prompts))

    elif file_name[-3:] == 'csv':
        df = pd.read_csv(file_name)
        prompts = copy.deepcopy(df['caption'].values)
        prompt_ids = copy.deepcopy(df.index.values)
        del df

    prompt_feature_dict = {}
    for i in tqdm(range(len(prompts))):
        prompt = tokenizer(prompts[i]).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = model.encode_text(prompt)

        prompt_feature_dict[prompt_ids[i]] = text_features.to('cpu').detach().numpy().copy()

    text_save_name = 'text_feature_' + task_name + '_' + args.model + '.pkl'
    print('Save text features in', text_save_name)
    with open(os.path.join(args.data_path, args.task, text_save_name), 'wb') as handle:
        pickle.dump(prompt_feature_dict, handle)

    print('Finish text feature extraction from', file_name)

if args.image_file is not None:
    file_name = os.path.join(args.data_path, args.task, args.image_file)
    print(f"reading images from {file_name}")

    image_files = os.listdir(file_name)
    image_feature_dict = {}
    for image_name in tqdm(image_files):
        img = preprocess(Image.open(os.path.join(file_name, image_name))).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(img)

        image_feature_dict[image_name[:-4]] = image_features.squeeze().detach().to('cpu').numpy().copy()

    image_save_name = 'image_feature_' + task_name + '_' + args.model + '.pkl'
    print('Save image features in', image_save_name)
    with open(os.path.join(args.data_path, args.task, image_save_name), 'wb') as handle:
        pickle.dump(image_feature_dict, handle)

    print('Finish image feature extraction from', file_name)


