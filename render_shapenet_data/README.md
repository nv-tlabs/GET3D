# Render ShapeNet Dataset
This script will help you render ShapeNet and custom datasets that follow ShapeNet conventions.

## Prerequisites
- Python 3.7 or higher
- Blender 2.9 or higher
- OSX, Linux or Windows running WSL2

## Setup
- Download the ShapeNet V1 or V2 dataset following the [official link](https://shapenet.org/)
- Make a new folder called shapenet in this directory, and unzip the downloaded file: `mkdir shapenet && unzip SHAPENET_SYNSET_ID.zip -d shapenet`
- Download Blender following the [official link](https://www.blender.org/)

## Installing Required Libraries
You will need the following libraries on Linux:
```
apt-get install -y libxi6 libgconf-2-4 libfontconfig1 libxrender1
```

Blender ships with its own distribution of Python, which you will need to add some libraries to:
```bash
cd BLENDER_PATH/2.90/python/bin
./python3.7m -m ensurepip
./python3.7m -m pip install numpy
```

## Data
The rendering script looks for datasets in the dataset_list.json file. You can modify this to add your own files and paths or point to your own JSON dataset list using the `--dataset_list <filename>` flag when invoking `render_all.py`

- For **ShapeNetCore.v1**, you don't need to do any preprocesing. If you are using your own dataset, you should make sure that your models are sorted into directories with a "model.obj" in them, following the expected conventions of ShapeNetCore.v1.

- For **ShapeNetCore.v2**, make sure to pass the `--shapenet_version 2` flag to the `render_all.py` script -- this will destructively normalize your dataset folder to match the expected structure of ShapeNetCore.v1, while retaining the original .obj and .mtl file names

## Rendering

### Quick Start
Once you've modified dataset_list.json or added the ShapeNet data that reflects the source training data paths, you can render your data like this:
```bash
python render_all.py
```

**Note:** The code will save the output from blender to `tmp.out`, this is not necessary for training, and can be removed by `rm -rf tmp.out`

## Additional Flags
You can customize the rendering script by adding flags.

### Switch to Eevee for dramatically faster rendering speed
By default, the Blender renderer uses Cycles, which has a photorealistic look but is slow (>10s/frame). You can also use Eevee, which may be more game-like in look but renders much much faster (<.3s/frame), and may be suitable for extracting a high quality dataset on lower end machines in a reasonable amount of time.
```
python render_all.py --engine EEVEE
```

### Render ShapeNet V2
For ShapeNetCore.v2 you will need to pass a flag to the render script to pre-process your data:
```bash
python render_all.py --shapenet_version 2
```

### Log to Console
By default, the script will log to a tmp.out file (quiet mode), but you can override this:
```
python render_all.py --quiet_mode 0

```
### Set Number of Views to Capture
The default for the rendering script is to capture 24 views per object. However, many NeRF pipeline recommend closer to 100 images. Especially if you are working with a limited but high quality dataset, you should consider increasing the total number of views 2-4x
```
python render_all.py --num_views 96
```

### Override Arguments
By default, the rendering script will save outputs to "shapenet_rendered", read all datasets from dataset_list.json and use the default Blender installation in your system. However, you can override these arguments:
```bash
python render_all.py --save_folder PATH_TO_SAVE_IMAGE --dataset_list PATH_TO_DATASET_JSON --blender_root PATH_TO_BLENDER
```

## Modifying the Render Scene
You can open the base scenes (located in the blender directory) and modify the lighting. There are no objects in the scene, so you will need to import a test object. Just be careful to remove any scene objects before you save.

## Comparison Between Cycles and Eevee
Cycles is on the **Left**, Eevee is on the **Right**
<br />
<img src="docs_img/cycles.png" alt="drawing" width="400"/> <img src="docs_img/eevee.png" alt="drawing" width="400"/>

To Render Eevee headlessly
```
!apt-get install python-opengl -y
!apt install xvfb -y
!pip install pyvirtualdisplay
!pip install piglet
python3 render_parallel.py --num_views 96 --engine EEVEE --headless
```

## Attribution

- This code is adopted from this [GitHub repo](https://github.com/panmari/stanford-shapenet-renderer), we thank the author for sharing the codes!

- The tome in the rendering comparison images was borrowed with permission from the [Loot Assets](https://github.com/webaverse/loot-assets) library.