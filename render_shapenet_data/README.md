## Render Shapenet Dataset

- Download shapenet V1 dataset following the [official link](https://shapenet.org/) and
  unzip the downloaded file `unzip SHAPENET_SYNSET_ID.zip`.
- Download Blender following the [official link](https://www.blender.org/), we used
  Blender **v2.90.0**, we haven't tested on other versions.
- Install required libraries:

```bash
apt-get install -y libxi6 libgconf-2-4 libfontconfig1 libxrender1
cd BLENDER_PATH/2.90/python/bin
./python3.7m -m ensurepip
./python3.7m -m pip install numpy 
```

- Running the render script:

```bash
python render_all.py --save_folder PATH_TO_SAVE_IMAGE --dataset_folder PATH_TO_3D_OBJ --blender_root PATH_TO_BLENDER
```

- (Optional) The code will save the output from blender to `tmp.out`, this is not
  necessary for training, and can be removed by `rm -rf tmp.out`


- This code is adopted from
  this [GitHub repo](https://github.com/panmari/stanford-shapenet-renderer), we thank the
  author for sharing the codes! 