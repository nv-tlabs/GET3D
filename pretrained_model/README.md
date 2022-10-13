

## Pretrained model

We released the pretrained model [here](https://drive.google.com/drive/folders/1oJ-FmyVYjIwBZKDAQ4N1EEcE9dJjumdW?usp=sharing) with the [License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

- Running inference only requires 1 GPU with >= 16GB GPU memory. 
- We provide the checkpoints for ShapeNet Car, Chair and Motorbikes that are trained on the `train.txt` subsets at `./3dgan_data_split`.
- We additionally provide the checkpoint for ShapeNet Table category, that is trained on all the objects of Table category. 
- We also proivde a Google Colab to try out our code [here](https://colab.research.google.com/drive/1AAE4jp39rXhW2zmlNwpWkvDPULugIXfk?usp=sharing).


### Inference Commands

We listed all the commands that can load the pretrained model here (working on Colab):

#### Generate 3D shapes and render it into 2D images w
```bash
python train_3d.py --outdir=save_inference_results/shapenet_car  --gpus=1 --batch=4 --gamma=40 --data_camera_mode shapenet_car  --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --resume_pretrain /content/GET3D/get3d_release/shapenet_car.pt
python train_3d.py --outdir=save_inference_results/shapenet_motorbike  --gpus=1 --batch=4 --gamma=40 --data_camera_mode shapenet_motorbike  --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --resume_pretrain /content/GET3D/get3d_release/shapenet_motorbike.pt
python train_3d.py --outdir=save_inference_results/shapenet_chair  --gpus=1 --batch=4 --gamma=400 --data_camera_mode shapenet_chair  --dmtet_scale 0.8  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --resume_pretrain /content/GET3D/get3d_release/shapenet_chair.pt
python train_3d.py --outdir=save_inference_results/shapenet_table  --gpus=1 --batch=4 --gamma=400 --data_camera_mode shapenet_chair  --dmtet_scale 0.8  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --resume_pretrain /content/GET3D/get3d_release/shapenet_table.pt
```

#### Generate 3D shapes and export the mesh with textures
```bash
python train_3d.py --outdir=save_inference_results/shapenet_car  --gpus=1 --batch=4 --gamma=40 --data_camera_mode shapenet_car  --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --resume_pretrain /content/GET3D/get3d_release/shapenet_car.pt --inference_to_generate_textured_mesh 1
python train_3d.py --outdir=save_inference_results/shapenet_motorbike  --gpus=1 --batch=4 --gamma=40 --data_camera_mode shapenet_motorbike  --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --resume_pretrain /content/GET3D/get3d_release/shapenet_motorbike.pt --inference_to_generate_textured_mesh 1
python train_3d.py --outdir=save_inference_results/shapenet_chair  --gpus=1 --batch=4 --gamma=400 --data_camera_mode shapenet_chair  --dmtet_scale 0.8  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --resume_pretrain /content/GET3D/get3d_release/shapenet_chair.pt --inference_to_generate_textured_mesh 1
python train_3d.py --outdir=save_inference_results/shapenet_table  --gpus=1 --batch=4 --gamma=400 --data_camera_mode shapenet_chair  --dmtet_scale 0.8  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --resume_pretrain /content/GET3D/get3d_release/shapenet_table.pt --inference_to_generate_textured_mesh 1
```