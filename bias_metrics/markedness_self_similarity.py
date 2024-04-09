import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import pickle
import argparse

ages = [
    "0-2",
    "9-Mar",
    "19-Oct",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "more than 70",
]  # csv converts to a date
age_labels = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "more than 70",
]  # to reconvert to age label
races = [
    "East Asian",
    "Indian",
    "Southeast Asian",
    "White",
    "Middle Eastern",
    "Latino_Hispanic",
    "Black",
]
genders = ["Female", "Male"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="RN50",
    help="model type in [RN50, ViT-B-32]",
)
parser.add_argument(
    "--from_pretrain",
    type=str,
    default="020",
    help="openai, cc12m, laion400m_e32, or an address to a pretrained model id (000, 020, etc.)",
)
parser.add_argument(
    "--data_path",
    type=str,
    help="Root path of the dataset",
)
args = parser.parse_args()

if args.from_pretrain in ["openai", "cc12m", "laion400m_e32"]:
    task_name = args.from_pretrain
else:
    task_name = "cc3m_mix_" + args.from_pretrain  # 'cc3m'

population_df = pd.read_csv(
    args.data_path + "FairFace/label_train.csv", index_col="file"
)
with open(
    args.data_path + "FairFace/image_feature_" + task_name + "_" + args.model + ".pkl",
    "rb",
) as handle:
    image_dict = pickle.load(handle)
image_df = pd.DataFrame(image_dict).T

# Set parameters for self-similarity measurements - smallest intersectional group has 22 images; 10k iterations
ITERS = 10000
MAX_ = 22

# Self-Similarity by Gender
means_ = []
gender_mean_dict = {}

for gender in genders:
    sub_df = population_df[population_df.gender.isin([gender])]
    sub_image = image_df.loc[sub_df.index.str.split("/").str[1].str.split(".").str[0]]
    img_arr = sub_image.to_numpy()

    self_sims = []

    for iter in range(ITERS):
        sub_arr = img_arr[np.random.randint(img_arr.shape[0], size=MAX_), :]
        sim = 1 - pdist(sub_arr, metric="cosine")
        self_sims.append(np.mean(sim))

    mean_ = np.mean(self_sims)
    means_.append(mean_)
    gender_mean_dict[gender] = mean_

print(genders)
print(means_)

intersection_df = pd.DataFrame(means_, index=genders, columns=["self_sim"])
# intersection_df.to_csv(f'D:\\gender_self_sims.csv')

# Self-Similarity by Race
means_ = []
race_mean_dict = {}

for race in races:
    sub_df = population_df[population_df.race.isin([race])]
    sub_image = image_df.loc[sub_df.index.str.split("/").str[1].str.split(".").str[0]]
    img_arr = sub_image.to_numpy()

    self_sims = []

    for iter in range(ITERS):
        sub_arr = img_arr[np.random.randint(img_arr.shape[0], size=MAX_), :]
        sim = 1 - pdist(sub_arr, metric="cosine")
        self_sims.append(np.mean(sim))

    mean_ = np.mean(self_sims)
    means_.append(mean_)
    race_mean_dict[race] = mean_

print(races)
print(means_)

intersection_df = pd.DataFrame(means_, index=races, columns=["self_sim"])
# intersection_df.to_csv(f'D:\\race_self_sims.csv')

# Self-Similarity by Age
means_ = []
age_mean_dict = {}

# for age in ages:
for age in age_labels:
    sub_df = population_df[population_df.age.isin([age])]
    sub_image = image_df.loc[sub_df.index.str.split("/").str[1].str.split(".").str[0]]
    img_arr = sub_image.to_numpy()

    self_sims = []

    for iter in range(ITERS):
        sub_arr = img_arr[np.random.randint(img_arr.shape[0], size=MAX_), :]
        sim = 1 - pdist(sub_arr, metric="cosine")
        self_sims.append(np.mean(sim))

    mean_ = np.mean(self_sims)
    means_.append(mean_)
    age_mean_dict[age] = mean_

print(age_labels)
print(means_)

intersection_df = pd.DataFrame(means_, index=age_labels, columns=["self_sim"])
# intersection_df.to_csv(f'D:\\age_self_sims.csv')


# Self-Similarity by Race and Age
population_df["intersection"] = population_df[["race", "age"]].agg(" ".join, axis=1)
intersections = list(set(population_df["intersection"].tolist()))

intersection_means = []
intersection_mean_dict = {}

for intersection in intersections:
    sub_df = population_df[population_df.intersection.isin([intersection])]
    sub_image = image_df.loc[sub_df.index.str.split("/").str[1].str.split(".").str[0]]
    img_arr = sub_image.to_numpy()

    self_sims = []

    for iter in range(ITERS):
        sub_arr = img_arr[np.random.randint(img_arr.shape[0], size=MAX_), :]
        sim = 1 - pdist(sub_arr, metric="cosine")
        self_sims.append(np.mean(sim))

    intersection_mean = np.mean(self_sims)
    intersection_means.append(intersection_mean)
    intersection_mean_dict[intersection] = intersection_mean

print(intersections)
print(intersection_means)

intersection_df = pd.DataFrame(
    intersection_means, index=intersections, columns=["self_sim"]
)
# intersection_df.to_csv(f'D:\\race_age_self_sims.csv')

# Self-Similarity by Gender and Age
wm_df_idx = population_df[population_df.gender.isin(["Male"])]
bf_df_idx = population_df[population_df.gender.isin(["Female"])]

age_count_list = []

# for age in ages:
for age in age_labels:
    sub_male = wm_df_idx[wm_df_idx.age.isin([age])]
    sub_female = bf_df_idx[bf_df_idx.age.isin([age])]

    age_count_list.append(
        len(sub_male.index.str.split("/").str[1].str.split(".").str[0].tolist())
    )
    age_count_list.append(
        len(sub_female.index.str.split("/").str[1].str.split(".").str[0].tolist())
    )

population_df["intersection"] = population_df[["gender", "age"]].agg(" ".join, axis=1)
intersections = list(set(population_df["intersection"].tolist()))

intersection_means = []
intersection_mean_dict = {}

for intersection in intersections:
    sub_df = population_df[population_df.intersection.isin([intersection])]
    sub_image = image_df.loc[sub_df.index.str.split("/").str[1].str.split(".").str[0]]
    img_arr = sub_image.to_numpy()

    self_sims = []

    for iter in range(ITERS):
        sub_arr = img_arr[np.random.randint(img_arr.shape[0], size=MAX_), :]
        sim = 1 - pdist(sub_arr, metric="cosine")
        self_sims.append(np.mean(sim))

    intersection_mean = np.mean(self_sims)
    intersection_means.append(intersection_mean)
    intersection_mean_dict[intersection] = intersection_mean

print(intersections)
print(intersection_means)

intersection_df = pd.DataFrame(
    intersection_means, index=intersections, columns=["self_sim"]
)
# intersection_df.to_csv(f'D:\\gender_age_self_sims.csv')


# Self-Similarity for All Intersections of Race, Gender, and Age
population_df["intersection"] = population_df[["race", "gender", "age"]].agg(
    " ".join, axis=1
)
intersections = list(set(population_df["intersection"].tolist()))

intersection_means = []
intersection_mean_dict = {}

for intersection in intersections:
    sub_df = population_df[population_df.intersection.isin([intersection])]
    sub_image = image_df.loc[sub_df.index.str.split("/").str[1].str.split(".").str[0]]
    img_arr = sub_image.to_numpy()

    self_sims = []

    for iter in range(ITERS):
        sub_arr = img_arr[np.random.randint(img_arr.shape[0], size=MAX_), :]
        sim = 1 - pdist(sub_arr, metric="cosine")
        self_sims.append(np.mean(sim))

    intersection_mean = np.mean(self_sims)
    intersection_means.append(intersection_mean)
    intersection_mean_dict[intersection] = intersection_mean

intersection_df = pd.DataFrame(
    intersection_means, index=intersections, columns=["self_sim"]
)
# intersection_df.to_csv(f'D:\\intersectional_self_sims.csv')

largest = intersection_df.nlargest(20, "self_sim")
smallest = intersection_df.nsmallest(20, "self_sim")

print(largest)
print(smallest)
