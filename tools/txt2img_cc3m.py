import numpy as np
import tarfile
import os
import glob
import io
import argparse


def random_seed(seed=42, rank=0):
    np.random.seed(seed)
    # random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=123, help="Default random seed.")
parser.add_argument(
    "--outdir",
    type=str,
    help="dir to write results to",
)
parser.add_argument(
    "--mix_ratio",
    type=float,
    default=0.2,
    help="mix ratio, in [0.2, 0.4, 0.6, 0.8, 1.0]",
)
parser.add_argument(
    "--cc3m_path",
    type=str,
    help="path of real cc3m tars",
)
parser.add_argument(
    "--sd_cc3m_path",
    type=str,
    help="path to generated cc3m images",
)
parser.add_argument(
    "--start_from",
    type=int,
    default=-1,
    help="starting point from broken process, default = 0",
)
args = parser.parse_args()

target_tars = glob.glob(args.cc3m_path + "*.tar")
target_tars.sort()

random_seed(args.seed)

print("=============== Mixing GCC with ratio", str(args.mix_ratio), "===============")

dst_path = args.outdir + str(int(args.mix_ratio * 100)).zfill(3) + "/"
os.makedirs(dst_path, exist_ok=True)

print("=============== Save files to", dst_path, "===============")

for tar_file in target_tars:
    if int(tar_file[-8:-4]) <= args.start_from:
        print("Skip", int(tar_file[-8:-4]), tar_file)
        continue

    print("=============== Loading from", tar_file, "===============")
    tar = tarfile.open(tar_file)
    files = tar.getnames()
    files.sort()

    mix_ids = np.random.choice(
        2, size=len(files) // 3, p=[1 - args.mix_ratio, args.mix_ratio]
    )  # 3 as 1 image always has its json and caption txt

    with tarfile.open(dst_path + tar_file.split("/")[-1], "w:gz") as archive:
        file_id = -1
        for i in range(len(files)):
            member = tar.getmember(files[i])
            file_id_ = int(files[i].split(".")[0])
            if not os.path.isfile(args.sd_cc3m_path + str(file_id_).zfill(8) + ".png"):
                print(str(file_id_).zfill(8) + ".png do not have human annotation.")
                continue

            if files[i][-3:] == "txt" or (
                files[i][-3:] in ["jpg", "png", "peg"] and mix_ids[file_id] != 1
            ):
                if files[i][-3:] == "txt":
                    print(file_id + 1, files[i][:-4])

                file = tar.extractfile(member)
                content = file.read()

                with io.BytesIO(content) as f:
                    info = tarfile.TarInfo(files[i])
                    f.seek(0, io.SEEK_END)
                    info.size = f.tell()
                    f.seek(0, io.SEEK_SET)
                    archive.addfile(info, f)

            elif files[i][-3:] in ["jpg", "png", "peg"] and mix_ids[file_id] == 1:
                file_id_ = int(files[i].split(".")[0])
                print(file_id + 1, files[i][:-4], str(file_id_).zfill(8) + ".png")
                archive.add(
                    args.sd_cc3m_path + str(file_id_).zfill(8) + ".png",
                    arcname=files[i],
                )

            file_id = i // 3

    archive.close()
    tar.close()
