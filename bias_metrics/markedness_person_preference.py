import open_clip
import numpy as np
import pandas as pd
import torch
import pickle
from clip_functions import balance_on_characteristic, balance_on_intersection, get_text_features
import argparse
import os

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
    help="The name of the task",
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
    help="Root path of the pre-trained model",
)
parser.add_argument(
    "--data_path",
    type=str,
    help="Root path of the dataset",
)
parser.add_argument(
    "--from_pretrain",
    type=str,
    default='020',
    help="openai, cc12m, laion400m_e32, or an address to a pre-trained model id (000, 020, etc.)",
)
parser.add_argument(
    "--output_root",
    type=str,
    help="Root path to save the output",
)
parser.add_argument(
    "--epoch",
    type=int,
    default=50,
    help="The number of epochs that the model is trained, default 50 (i.e., epoch_XX.pt)",
)
args = parser.parse_args()

if args.from_pretrain in ['openai','cc12m', 'laion400m_e32']:
    task_name = args.from_pretrain
    model_path = args.from_pretrain
else:
    task_name = 'cc3m_mix_' + args.from_pretrain # 'cc3m'
    model_path = os.path.join(args.model_path + 'cc3m_mix_' + args.from_pretrain, 'epoch_'+str(args.epoch)+'.pt')

#Initialize CLIP
print('Initialize CLIP')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model, _, preprocess = open_clip.create_model_and_transforms(
    args.model,
    pretrained=model_path,
    device=device,
)
tokenizer = open_clip.get_tokenizer(args.model)
model.eval()

#Read in FairFace labels and projected image embeddings; normalize image embeddings for cosine similarity measurements
print('Read in FairFace labels and projected image embeddings')
population_df = pd.read_csv(args.data_path + 'FairFace/label_train.csv',index_col='file')

with open(args.data_path + 'FairFace/image_feature_' + task_name + '_' + args.model + '.pkl', 'rb') as handle:
    image_dict = pickle.load(handle)
image_features = np.array(list(image_dict.values()))
print(image_features.shape)
image_features = image_features / np.linalg.norm(image_features,axis=1,keepdims=True)
image_list = list(image_dict.keys())

output_root = args.output_root

#Labels used in dataframes, based on FairFace labels
ages = ['0-2','9-Mar','19-Oct','20-29','30-39','40-49','50-59','60-69','more than 70']
age_labels = ['0-2','3-9','10-19','20-29','30-39','40-49','50-59','60-69','more than 70']
races = ['East Asian','Indian','Southeast Asian','White','Middle Eastern','Latino_Hispanic','Black']
genders = ['Female','Male']

#Define prompts for age, race or ethnicity, and gender
age_tuples = [('0','2'),('3','9'),('10','19'),('20','29'),('30','39'),('40','49'),('50','59'),('60','69')]
ages_ = [f'a photo of a person between {tup[0]} and {tup[1]} years old' for tup in age_tuples]
ages_.append('a photo of a person 70 or more years old')

races_ = [f'a photo of {race_} person' for race_ in ['an East Asian','an Indian','a Southeast Asian','a White','a Middle Eastern','a Latino or Hispanic','a Black']]
genders_ = [f'a photo of a female person','a photo of a male person']

#Race or Ethnicity similarities
print('Race or Ethnicity similarities')
race_similarities = []

#Similarity with "person"
with torch.no_grad():
    text_features = get_text_features(['a photo of a person'],model,tokenizer).cpu().numpy().squeeze()
    text_features = text_features / np.linalg.norm(text_features)

prompt_similarities = image_features @ text_features.T
race_similarities.append(prompt_similarities)

#Cosine similarities with each race or ethnicity prompt
for prompt in races_:
    
    with torch.no_grad():
        text_features = get_text_features([prompt],model,tokenizer).cpu().numpy().squeeze()
        text_features = text_features / np.linalg.norm(text_features)
    
    prompt_similarities = image_features @ text_features.T
    race_similarities.append(prompt_similarities)

#Dataframe of similarities
race_df = pd.DataFrame(np.array(race_similarities).T,index=image_list,columns=['Person'] + races)
race_df.to_csv(output_root + f'race_similarities_' + task_name + '_' + args.model + '.csv')

#Determine which images CLIP leaves unlabeled based on race or ethnicity
race_preferences = race_df[race_df.columns].idxmax(axis=1).tolist()
binary_race_preferences = [1 if pref == 'Person' else 0 for pref in race_preferences]
race_df = pd.DataFrame(binary_race_preferences,index=image_list,columns=['Race'])

#Gender
print('Gender')
gender_similarities = []

#Person similarities
with torch.no_grad():
    text_features = get_text_features(['a photo of a person'],model,tokenizer).cpu().numpy().squeeze()
    text_features = text_features / np.linalg.norm(text_features)

prompt_similarities = image_features @ text_features.T
gender_similarities.append(prompt_similarities)

#Similarities for each gender prompt
for prompt in genders_:
    
    with torch.no_grad():
        text_features = get_text_features([prompt],model,tokenizer).cpu().numpy().squeeze()
        text_features = text_features / np.linalg.norm(text_features)
    
    prompt_similarities = image_features @ text_features.T
    gender_similarities.append(prompt_similarities)

#Dataframe of similarities by gender
gender_df = pd.DataFrame(np.array(gender_similarities).T,index=image_list,columns=['Person'] + genders)
gender_df.to_csv(output_root + f'gender_similarities_' + task_name + '_' + args.model + '.csv')

#Determine which images CLIP leaves unlabeled based on gender
gender_preferences = gender_df[gender_df.columns].idxmax(axis=1).tolist()
binary_gender_preferences = [1 if pref == 'Person' else 0 for pref in gender_preferences]
gender_df = pd.DataFrame(binary_gender_preferences,index=image_list,columns=['Gender'])

#Age
print('Age')
age_similarities = []

#Person Similarities
with torch.no_grad():
    text_features = get_text_features(['a photo of a person'],model,tokenizer).cpu().numpy().squeeze()
    text_features = text_features / np.linalg.norm(text_features)

prompt_similarities = image_features @ text_features.T
age_similarities.append(prompt_similarities)

#Similarities for each age prompt
for prompt in ages_:
    
    with torch.no_grad():
        text_features = get_text_features([prompt],model,tokenizer).cpu().numpy().squeeze()
        text_features = text_features / np.linalg.norm(text_features)
    
    prompt_similarities = image_features @ text_features.T
    age_similarities.append(prompt_similarities)

#Dataframe of similarities by age
age_df = pd.DataFrame(np.array(age_similarities).T,index=image_list,columns=['Person'] + ages)
age_df.to_csv(output_root + f'age_similarities_' + task_name + '_' + args.model + '.csv')

#Determine which images CLIP leaves unlabeled based on age
age_preferences = age_df[age_df.columns].idxmax(axis=1).tolist()
binary_age_preferences = [1 if pref == 'Person' else 0 for pref in age_preferences]
age_df = pd.DataFrame(binary_age_preferences,index=image_list,columns=['Age'])

#Concatenate race, gender, age dataframes
print('Concatenate race, gender, age dataframes')
joint_df = pd.concat([race_df,gender_df,age_df],axis=1)
joint_df.to_csv(output_root + f'clip_preferences_' + task_name + '_' + args.model + '.csv')
print(joint_df)

#Downsample to equalize based on frequency in the dataset
ITERS = 10000

#Test of CLIP's preference to leave race unobserved
pref_dict = {race:[] for race in races}

for iter in range(ITERS):
    for race in races:
        balanced_df, balanced_preferences = balance_on_characteristic(population_df,joint_df,'race') #Balance dataset
        race_idx = balanced_df[balanced_df.race.isin([race])] #Sample only from target race or ethnicity
        prefs = balanced_preferences.loc[race_idx.index.str.split('/').str[1].str.split('.').str[0]]
        pref_list = prefs['Race'].tolist()
        pct = sum(pref_list)/len(pref_list) #Get percentage of images for which "person" is highest probability
        pref_dict[race].append(pct)

mean_dict = {}

#Return mean percentage of the time for which "person" is highest probability for each race or ethnicity
for race in races:
    race_mean = np.mean(pref_dict[race])
    mean_dict[race] = race_mean

print(mean_dict)

#Test of CLIP's preference to leave gender unobserved
pref_dict = {gender:[] for gender in genders}

for iter in range(ITERS):
    for gender in genders:
        balanced_df, balanced_preferences = balance_on_characteristic(population_df,joint_df,'gender') #Balance dataset
        gender_idx = balanced_df[balanced_df.gender.isin([gender])] #Sample only from target gender
        prefs = balanced_preferences.loc[gender_idx.index.str.split('/').str[1].str.split('.').str[0]]
        pref_list = prefs['Gender'].tolist()
        pct = sum(pref_list)/len(pref_list) #Get percentage of images for which "person" is highest probability
        pref_dict[gender].append(pct)

mean_dict = {}

#Return mean percentage of the time for which "person" is highest probability for each gender
for gender in genders:
    gender_mean = np.mean(pref_dict[gender])
    mean_dict[gender] = gender_mean

print(mean_dict)

#Test of CLIP's preference to leave age unobserved
pref_dict = {age:[] for age in age_labels}

for iter in range(ITERS):
    for age in age_labels:
        balanced_df, balanced_preferences = balance_on_characteristic(population_df,joint_df,'age') #Balance dataset
        balanced_df, balanced_preferences = balance_on_characteristic(population_df,joint_df,'age') #Balance dataset
        age_idx = balanced_df[balanced_df.age.isin([age])] #Sample only from target age
        prefs = balanced_preferences.loc[age_idx.index.str.split('/').str[1].str.split('.').str[0]]
        pref_list = prefs['Age'].tolist()
        pct = sum(pref_list)/len(pref_list) #Get percentage of images for which "person" is highest probability
        pref_dict[age].append(pct)

mean_dict = {}

#Return mean percentage of the time for which "person" is highest probability for each age
for age in age_labels:
    age_mean = np.mean(pref_dict[age])
    mean_dict[age] = age_mean

print(mean_dict)

#Examination of markedness by intersection of gender and age range
balanced_df, balanced_preferences = balance_on_intersection(population_df,joint_df,['gender','age'])
intersections = list(set(balanced_df['intersection'].tolist()))
pref_dict_age,pref_dict_gender,pref_dict_race = {intersection:[] for intersection in intersections},{intersection:[] for intersection in intersections},{intersection:[] for intersection in intersections}

for iter in range(ITERS):
    for intersection in intersections:

        #Balance for gender and age and return preference for target intersection
        balanced_df, balanced_preferences = balance_on_intersection(population_df,joint_df,['gender','age'])
        intersection_idx = balanced_df[balanced_df.intersection.isin([intersection])]
        prefs = balanced_preferences.loc[intersection_idx.index.str.split('/').str[1].str.split('.').str[0]]

        #Get preference to leave age, gender, race unobserved
        pref_list_age = prefs['Age'].tolist()
        pref_list_gender = prefs['Gender'].tolist()
        pref_list_race = prefs['Race'].tolist()

        #Get percentage of the time unobserved
        pct_age = sum(pref_list_age)/len(pref_list_age)
        pct_gender = sum(pref_list_gender)/len(pref_list_gender)
        pct_race = sum(pref_list_race)/len(pref_list_race)
        
        pref_dict_age[intersection].append(pct_age)
        pref_dict_gender[intersection].append(pct_gender)
        pref_dict_race[intersection].append(pct_race)

mean_dict_age,mean_dict_gender,mean_dict_race = {},{},{}

#Get mean percentage of the time age, gender, and race is left unobserved
for intersection in intersections:
    age_mean = np.mean(pref_dict_age[intersection])
    mean_dict_age[intersection] = age_mean

    gender_mean = np.mean(pref_dict_gender[intersection])
    mean_dict_gender[intersection] = gender_mean

    race_mean = np.mean(pref_dict_race[intersection])
    mean_dict_race[intersection] = race_mean

male_age_series = [mean_dict_age[f'Male {age}'] for age in age_labels]
female_age_series = [mean_dict_age[f'Female {age}'] for age in age_labels]

male_gender_series = [mean_dict_gender[f'Male {age}'] for age in age_labels]
female_gender_series = [mean_dict_gender[f'Female {age}'] for age in age_labels]

male_race_series = [mean_dict_race[f'Male {age}'] for age in age_labels]
female_race_series = [mean_dict_race[f'Female {age}'] for age in age_labels]

#Print to output
print(male_age_series)
print(female_age_series)

print(male_gender_series)
print(female_gender_series)

print(male_race_series)
print(female_race_series)

print(' '.join([f'({idx+1},{male_age_series[idx]})' for idx in range(len(age_labels))]))
print(' '.join([f'({idx+1},{female_age_series[idx]})' for idx in range(len(age_labels))]))

print(' '.join([f'({idx+1},{male_gender_series[idx]})' for idx in range(len(age_labels))]))
print(' '.join([f'({idx+1},{female_gender_series[idx]})' for idx in range(len(age_labels))]))

print(' '.join([f'({idx+1},{male_race_series[idx]})' for idx in range(len(age_labels))]))
print(' '.join([f'({idx+1},{female_race_series[idx]})' for idx in range(len(age_labels))]))

#Direct comparison of intersectional disparities based on race, gender, and age
ITERS = 10000

balanced_df, balanced_preferences = balance_on_intersection(population_df,joint_df,['race','gender','age'])
intersections = list(set(balanced_df['intersection'].tolist()))
pref_dict_age,pref_dict_gender,pref_dict_race = {intersection:[] for intersection in intersections},{intersection:[] for intersection in intersections},{intersection:[] for intersection in intersections}

wm_df_idx = population_df[population_df.race.isin(['White']) & population_df.gender.isin(['Male'])]
bf_df_idx = population_df[population_df.race.isin(['Black']) & population_df.gender.isin(['Female'])]

MAX_ = 22

#Dictionaries to hold lists for each age range
wm_mean_dict = {'Race': [], 'Gender': [], 'Age': []}
bf_mean_dict = {'Race': [], 'Gender': [], 'Age': []}

for age in age_labels:

    #Select current age range
    wm_df = wm_df_idx[wm_df_idx.age.isin([age])]
    bf_df = bf_df_idx[bf_df_idx.age.isin([age])]

    wm_sub_df = joint_df.loc[wm_df.index.str.split('/').str[1].str.split('.').str[0]]
    bf_sub_df = joint_df.loc[bf_df.index.str.split('/').str[1].str.split('.').str[0]]

    wm_count_dict = {'Race': [], 'Gender': [], 'Age': []}
    bf_count_dict = {'Race': [], 'Gender': [], 'Age': []}

    for iter in range(ITERS):
        #Sample subset of each intersectional group for the age range
        wm_sample = wm_sub_df.sample(MAX_)
        bf_sample = bf_sub_df.sample(MAX_)

        #Append counts to sample lists
        wm_count_dict['Race'].append(sum(wm_sample['Race'].tolist()))
        wm_count_dict['Gender'].append(sum(wm_sample['Gender'].tolist()))
        wm_count_dict['Age'].append(sum(wm_sample['Age'].tolist()))

        bf_count_dict['Race'].append(sum(bf_sample['Race'].tolist()))
        bf_count_dict['Gender'].append(sum(bf_sample['Gender'].tolist()))
        bf_count_dict['Age'].append(sum(bf_sample['Age'].tolist()))

    #Get the mean counts at the current age range
    wm_mean_dict['Race'].append(np.mean(wm_count_dict['Race'])/MAX_)
    wm_mean_dict['Gender'].append(np.mean(wm_count_dict['Gender'])/MAX_)
    wm_mean_dict['Age'].append(np.mean(wm_count_dict['Age'])/MAX_)

    bf_mean_dict['Race'].append(np.mean(bf_count_dict['Race'])/MAX_)
    bf_mean_dict['Gender'].append(np.mean(bf_count_dict['Gender'])/MAX_)
    bf_mean_dict['Age'].append(np.mean(bf_count_dict['Age'])/MAX_)

#Print results
print(wm_mean_dict)
print(bf_mean_dict)

print(' '.join([f'({idx+1},{wm_mean_dict["Race"][idx]*100})' for idx in range(len(age_labels))]))
print(' '.join([f'({idx+1},{bf_mean_dict["Race"][idx]*100})' for idx in range(len(age_labels))]))

print(' '.join([f'({idx+1},{wm_mean_dict["Gender"][idx]*100})' for idx in range(len(age_labels))]))
print(' '.join([f'({idx+1},{bf_mean_dict["Gender"][idx]*100})' for idx in range(len(age_labels))]))

print(' '.join([f'({idx+1},{wm_mean_dict["Age"][idx]*100})' for idx in range(len(age_labels))]))
print(' '.join([f'({idx+1},{bf_mean_dict["Age"][idx]*100})' for idx in range(len(age_labels))]))
