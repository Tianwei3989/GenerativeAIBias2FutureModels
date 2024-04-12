from torch import autocast
import torch
from diffusers import StableDiffusionPipeline
from pytorch_lightning import seed_everything
import argparse
from tqdm import tqdm
from itertools import islice
import os

parser = argparse.ArgumentParser()


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


parser.add_argument(
    "--outdir",
    type=str,
    help="dir to write results to",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=8,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--from_file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--from_file_id",
    type=str,
    help="if specified, load prompts from this file id",
)
parser.add_argument(
    "--seed",
    type=int,
    default=123,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--start",
    type=int,
    help="Enable to start from a breakpoint (in case of process failure)",
)
opt = parser.parse_args()

seed_everything(opt.seed)
batch_size = opt.n_samples

os.makedirs(opt.outdir, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    safety_checker=None,
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

# read file by id if exist
if not opt.from_file_id:
    file_name = opt.from_file
    print("No file ID!")
    print()
else:
    file_name = opt.from_file.split(".")[0] + "_" + opt.from_file_id + ".txt"

print(f"reading prompts from {file_name}")
with open(file_name, "r") as f:
    data = f.read().splitlines()
    data = list(chunk(data, batch_size))

for texts in tqdm(data, desc="data"):
    # divide tid and prompts
    tids = []
    prompts = []
    for text_count_id in range(len(texts)):
        t_name = texts[text_count_id].split("#")[0].zfill(8)
        if int(t_name) < opt.start:
            print("Skip", t_name, ": to reach the new start point:", opt.start)
            continue
        if os.path.isfile(
            os.path.join(opt.outdir + "_" + str(opt.seed), t_name + ".png")
        ):
            print("File", t_name, "exist, skip the generation.")
            continue
        tids.append(t_name)
        prompts.append("#".join(texts[text_count_id].split("#")[1:]))
        t_count = 0

    if len(tids) == 0:
        continue

    with autocast("cuda"):
        images = pipe(prompts)

    for i in range(len(prompts)):
        image = images[0][i]
        image.save(os.path.join(opt.outdir + "_" + str(opt.seed), tids[i] + ".png"))
        print(tids[i], prompts[i])
