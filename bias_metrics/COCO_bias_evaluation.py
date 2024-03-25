import pickle
import json
import numpy as np
import pandas as pd
import os
import torch
import shutil
import argparse

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res

def get_index_list(df_anns, attr_type, pick_attr):
    '''get the index of subgroup in all the val set'''
    index_list = []
    image_ids = []
    for index, i in enumerate(df_anns.id):
        if attr_type == 'bb_gender':
            attr_list = [df_anns.iloc[index][attr_type]]
        elif attr_type == 'skin':
            attr_list = df_anns.iloc[index][attr_type].strip("']['").split("', '")

        if len(set(attr_list)) == 1 and attr_list[0] == pick_attr:
            samples = df_bias_[df_bias_['image_id']==i]
            index_list += samples.index.tolist()

            image_ids.append(index)

    print(attr_type, pick_attr)
    print('caption size', len(index_list), 'image size', len(image_ids))
        
    return index_list

def compute_attr_acc(similarity, index_list):
    index_tensor = torch.tensor(np.array(labels)[index_list])
    sim_mtx = similarity[index_list]
    print('sim_mtx.shape', sim_mtx.shape[0])
    r1, r5, r10 = accuracy(sim_mtx, index_tensor, topk=(1, 5, 10))
    
    return r1, r5, r10

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    help="path to PHASE dataset",
)
parser.add_argument(
    "--feature_root",
    type=str,
    help="path to PHASE image feature extracted by trained OpenCLIP",
)
parser.add_argument(
    "--pre_train",
    type=str,
    help="name of the model, e.g., openai, laion400m_e32, cc12m, cc3m_mix_020",
)
parser.add_argument(
    "--model_name",
    type=str,
    help="model framewrk name, e.g., ViT-B-32 and RN50",
)
args = parser.parse_args()


bias_dict = {
    'bb_gender':['Male','Female'],
    'skin':['Light','Dark'],
}

df_img = pd.read_csv(os.path.join(args.data_root, 'images_val2014.csv'))
df_img_val = df_img[df_img['split']=='val']

anns_path = os.path.join(args.data_root, 'annotations/captions_val2014.json')
with open(anns_path, 'r') as j:
     anns = json.loads(j.read())

df_caption = pd.DataFrame(anns['annotations'])

dst_img_path = os.path.join(args.data_root, 'images/val/')
os.makedirs(dst_img_path, exist_ok=True)

df_bias = pd.DataFrame()
for i in df_img_val.id.values:
    sample = df_caption[df_caption['image_id'] == i]
    if sample.shape[0] == 0:
        print(i)
    df_bias = pd.concat([df_bias, sample])

image_names = []
for i in df_bias['image_id']:
    image_names.append('COCO_val2014_'+str(i).zfill(12)+'.jpg')
df_bias['image_name'] = image_names
df_bias_ = df_bias.reset_index()

# prepare labels
labels = []
label_raw = df_bias_['image_id'].unique().tolist()
label_raw.sort()
for i in range(len(label_raw)):
    samples = df_bias_[df_bias_['image_id']==label_raw[i]]
    labels += [i] * samples.shape[0]

# evaluation
img_pickle_path = os.path.join(args.data_root,'images/IR_img_feature_coco_bias_'+args.pre_train+'_'+args.model_name+'.pkl')
with open(img_pickle_path, 'rb') as handle:
    img_feature = pickle.load(handle)
    
cap_pickle_path = os.path.join(args.data_root,'images/IR_txt_feature_coco_bias_'+args.pre_train+'_'+args.model_name+'.pkl')
with open(cap_pickle_path, 'rb') as handle:
    cap_feature = pickle.load(handle)


img_feature_copy = torch.tensor(np.array(list(img_feature.values()))).float()
cap_feature_copy = torch.tensor(np.array(list(cap_feature.values()))).float()

img_feature_copy /= img_feature_copy[1].norm(dim=-1, keepdim=True)
cap_feature_copy /= cap_feature_copy.norm(dim=-1, keepdim=True)
similarity = (100.0 * img_feature_copy @ cap_feature_copy.T).T

# global
r1, r5, r10 = compute_attr_acc(similarity, np.arange(similarity.shape[0]))
print('Global result')
print("R@1:",r1,"R@5:",r5,"R@10:",r10)
print()

# By attributes
for k in bias_dict.keys():
    categories = bias_dict[k]
    for c in categories:
        index_list = get_index_list(df_img_val, k, c)
        r1, r5, r10 = compute_attr_acc(similarity, index_list)
        print('Results on', c, 'in', k)
        print("R@1:",r1,"R@5:",r5,"R@10:",r10)
        print()
