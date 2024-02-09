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
import pickle
from tqdm import tqdm

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
    "--task",
    type=str,
    default='flickr',
    help="in [flickr, coco]",
)
parser.add_argument(
    "--model",
    type=str,
    default='RN50',
    help="model type in [RN50, ViT-B-32]",
)
parser.add_argument(
    "--text_file",
    type=str,
    default='final_test_set0_2014.jsonline',
    help="read jsonline files of captions and image index",
)
parser.add_argument(
    "--data_root",
    type=str,
    default='/home/chentw/Dataset/clip_bias/IR_test/',
    help="The address of dataset",
)
parser.add_argument(
    "--from_pretrain",
    type=str,
    # default='/home/chentw/work/open_clip/src/logs/2023_09_20-05_47_53-model_RN50-lr_0.0005-b_360-j_4-p_amp/checkpoints/epoch_50.pt',
    default='020',
    help="openai, cc12m, laion400m_e32, or an address to a pretrained model id (000, 020, etc.)",
)
parser.add_argument(
    "--epoch",
    type=int,
    default=32,
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
    model_path = os.path.join('/home/chentw/work/open_clip/models/cc3m_mix_' + args.from_pretrain, 'epoch_'+str(args.epoch)+'.pt')

random_seed(args.seed, 0)

# read file by id if exist
data_root = args.data_root

# image_files = os.listdir(data_root + args.task + '/')
# image_files.sort()

text_file = data_root + args.task + '_' + args.text_file
df_text = pd.read_json(text_file, lines=True)
if args.task == 'flickr':
    prompts = copy.deepcopy(df_text['sentences'].values.sum())
    image_ids = copy.deepcopy(df_text['img_path'].values.tolist())
else:
    prompts = copy.deepcopy(df_text['sentences'].values.tolist())
    image_ids = copy.deepcopy(df_text['img_path'].str.split('/').str[-1].values.tolist())
del df_text

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Load model from', args.from_pretrain)
model, _, preprocess = open_clip.create_model_and_transforms(
    args.model,
    pretrained=model_path,
    device=device,
)
tokenizer = open_clip.get_tokenizer(args.model)

print('Processing text')
feature_len = 1024 if args.model == 'RN50' else 512
out_txt_features = {}
# out_txt_all = torch.empty((0, feature_len), device=device)
for i in tqdm(range(len(prompts))):
    text_sample = prompts[i]
    text = tokenizer(text_sample).to(device)
    # text = tokenizer(prompts[:5])

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        # text_features /= text_features.norm(dim=-1, keepdim=True)

        # out_txt_all = torch.vstack((out_txt_all, text_features))
        out_txt_features[i] = text_features.squeeze().cpu().numpy()
        # print(i)

save_pickle(data_root + 'IR_txt_feature_'+args.task+'_'+task_name+'_'+args.model+'.pkl', out_txt_features)  # fake for test
del out_txt_features

print('Ranking images')
out_img_features = {}
# out_img_all = torch.empty((0, feature_len), device=device)
for i in tqdm(range(len(image_ids))):
    image_file = os.path.join(data_root, args.task, str(image_ids[i]))
    image = preprocess(Image.open(image_file)).to(device).unsqueeze(0)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        # image_features_ /= image_features.norm(dim=-1, keepdim=True)

        # out_img_all = torch.vstack((out_img_all, image_features))
        out_img_features[image_ids[i]] = image_features.squeeze().cpu().numpy()
        # print(i)

save_pickle(data_root + 'IR_img_feature_'+args.task+'_'+task_name+'_'+args.model+'.pkl', out_img_features)