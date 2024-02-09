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

def save_pickle(pickle_path, feature):
    with open(pickle_path, 'wb') as handle:
        pickle.dump(feature, handle)
    
    print('pickle saved to', pickle_path)

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
    "--model_path",
    type=str,
    help="Root path of the pretrained model",
)
parser.add_argument(
    "--data_path",
    type=str,
    help="Root path of the dataset",
)
parser.add_argument(
    "--task",
    type=str,
    default='coco_bias',
    help="in [PHASE, coco_bias]",
)
parser.add_argument(
    "--model",
    type=str,
    default='RN50',
    help="model type in [RN50, ViT-B-32]",
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

# read file by id if exist
data_root = args.data_path + args.task+'/'

image_files = os.listdir(data_root + 'images/val/')

if args.task == 'PHASE':
    text_file = args.data_path + 'PHASE/phase_annotations/phase_gcc_val_regions_20221101.json'
    df_text = pd.read_json(text_file).T.reset_index()
    prompts = copy.deepcopy(df_text['caption'].values.tolist())
    image_ids = copy.deepcopy(df_text['index'].values.tolist())
else:
    text_file = args.data_path + args.task+'/annotation.json'
    df_text = pd.read_json(text_file)
    prompts = copy.deepcopy(df_text['caption'].values.tolist())
    image_ids = copy.deepcopy(df_text['image_name'].values.tolist())
del df_text

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Load model from', model_path)
model, _, preprocess = open_clip.create_model_and_transforms(
    args.model,
    pretrained=model_path,
    device=device,
)
tokenizer = open_clip.get_tokenizer(args.model)

print('Processing text')
feature_len = 1024 if args.model == 'RN50' else 512
out_txt_features = {}
for i in tqdm(range(len(prompts))):
    text_sample = prompts[i]
    text = tokenizer(text_sample).to(device)
    # text = tokenizer(prompts[:5])

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        out_txt_features[i] = text_features.squeeze().cpu().numpy()

save_pickle(data_root + 'images/IR_txt_feature_'+args.task+'_'+task_name+'_'+str(args.epoch)+'_'+args.model+'.pkl', out_txt_features)
del out_txt_features

print('Ranking images')
out_img_features = {}
for i in tqdm(range(len(image_ids))):
    image_file = os.path.join(data_root + 'images/val/', str(image_ids[i]))
    image = preprocess(Image.open(image_file)).to(device).unsqueeze(0)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        out_img_features[image_ids[i]] = image_features.squeeze().cpu().numpy()

save_pickle(data_root + 'images/IR_img_feature_'+args.task+'_'+task_name+'_'+str(args.epoch)+'_'+args.model+'.pkl', out_img_features)
