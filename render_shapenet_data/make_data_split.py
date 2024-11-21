# This script generates data manifests for your dataset, divided into training, testing and evaluation lists

# args:
# sys.argv[1] = name or id of the synset
# sys.argv[2] = directory of the synset to generate from (should be a list of directories)
# sys.argv[3] = output directory (optional, defaults to "./3dgan_data_split")

import os
import sys
import random

# the second arg is the source directory
source_dir = sys.argv[2]

# the first arg is the name, if it is null then set it to the last part of the source directory
name = sys.argv[1] if len(sys.argv) > 2 else source_dir.split("/")[-1]

# default output directory is 3dgan_data_split, but you can specify a different one
data_dir = "3dgan_data_split"

if len(sys.argv) > 3:
    data_dir = sys.argv[3]

os.makedirs(data_dir, exist_ok=True)

os.makedirs(os.path.join(data_dir, name), exist_ok=True)

all = []

for folder in os.listdir(source_dir):
    all.append(folder)

train = [] # 70%
test = [] # 20%
val = [] # 10%

random.shuffle(all)
train = all[:int(len(all) * 0.7)]
test = all[int(len(all) * 0.7):int(len(all) * 0.9)]
val = all[int(len(all) * 0.9):]

with open(os.path.join(data_dir, name, "train.txt"), "w") as f:
    f.write("\n".join(train))
with open(os.path.join(data_dir, name, "test.txt"), "w") as f:
    f.write("\n".join(test))
with open(os.path.join(data_dir, name, "val.txt"), "w") as f:
    f.write("\n".join(val))

print ("Finished");