import numpy as np
import json
import pickle
import torch
import os
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

def get_index_list(anns, attr_type, pick_attr):
    '''get the index of subgroup in all the val set'''
    index_list = []
    for index, i in enumerate(anns.keys()):
        anns_key = anns[i].keys()
        attr_list = []
        for a in anns_key:
            if a != 'caption':
                attr = anns[i][a]['annotations'][attr_type]
                attr_list.append(attr)
        assert len(attr_list) + 1 == len(anns_key)
        attr_list = list(set(attr_list))
        if len(attr_list) == 1 and attr_list[0] == pick_attr:
            index_list.append(index)
    
    print(attr_type, pick_attr)
    print('size', len(index_list))
    return index_list

def compute_attr_acc(similarity, index_list):
    index_tensor = torch.tensor(index_list)
    sim_mtx = similarity[index_list]
    r1, r5, r10 = accuracy(sim_mtx, index_tensor, topk=(1, 5, 10))
    return r1.cpu().numpy(), r5.cpu().numpy(), r10.cpu().numpy()

def get_baby_child_list(anns, attr_type):
    '''get the index of subgroup in all the val set'''
    index_list = []
    for index, i in enumerate(anns.keys()):
        anns_key = anns[i].keys()
        attr_list = []
        for a in anns_key:
            if a != 'caption':
                attr = anns[i][a]['annotations'][attr_type]
                attr_list.append(attr)
        assert len(attr_list) + 1 == len(anns_key)
        attr_list = list(set(attr_list))
        if len(attr_list) == 1:
            if attr_list[0] in ['baby', 'children']:
                index_list.append(index)
    
    print(attr_type)
    print('size', len(index_list))
    return index_list

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
    'age':['baby&child','young','adult','senior'],
    'gender':['male','female'],
    'skintone':['lighter','darker'],
    'ethnicity':['black','east_asian','indian','latino','middle_eastern','southeast_asian','white'],
}


anns_path = os.path.join(args.data_root,'anns_imgs_gcc_val_majorities_regions_20221101.json')
with open(anns_path, 'r') as j:
     anns = json.loads(j.read())

img_pickle_path = os.path.join(args.feature_root, 'IR_img_feature_PHASE_'+args.pre_train+'_'+args.model_name+'.pkl')
with open(img_pickle_path, 'rb') as handle:
    img_feature = pickle.load(handle)
    
cap_pickle_path = os.path.join(args.feature_root, 'IR_txt_feature_PHASE_'+args.pre_train+'_'+args.model_name+'.pkl')
with open(cap_pickle_path, 'rb') as handle:
    cap_feature = pickle.load(handle)


img_feature_copy = torch.tensor(np.array(list(img_feature.values()))).float()
cap_feature_copy = torch.tensor(np.array(list(cap_feature.values()))).float()

img_feature_copy /= img_feature_copy[1].norm(dim=-1, keepdim=True)
cap_feature_copy /= cap_feature_copy.norm(dim=-1, keepdim=True)
similarity = (100.0 * img_feature_copy @ cap_feature_copy.T).T#.softmax(dim=-1)

# global
r1, r5, r10 = compute_attr_acc(similarity, np.arange(similarity.shape[0]))
print('Global result')
print("R@1:",r1,"R@5:",r5,"R@10:",r10)
print()

# By attributes
for k in bias_dict.keys():
    categories = bias_dict[k]
    for c in categories:
        if c == 'baby&child':
            index_list = get_baby_child_list(anns, 'age')
        else:
            index_list = get_index_list(anns, k, c)
        
        r1, r5, r10 = compute_attr_acc(similarity, index_list)
        print('Results on', c, 'in', k)
        print("R@1:",r1,"R@5:",r5,"R@10:",r10)
        print()






























