# BoosterSHOT
Repository containing code for BoosterSHOT

## Abstract
Improving multi-view aggregation is integral for multi-view pedestrian detection, which aims to obtain a bird's-eye-view pedestrian occupancy map from images captured through a set of calibrated cameras. Inspired by the success of attention modules for deep neural networks, we first propose a Homography Attention Module (HAM) which is shown to boost the performance of existing end-to-end multiview detection approaches by utilizing a novel channel gate and spatial gate. Additionally, we propose Booster-SHOT, an end-to-end convolutional approach to multiview pedestrian detection incorporating our proposed HAM as well as elements from previous approaches such as view-coherent augmentation or stacked homography transformations. Booster-SHOT achieves 92.9% and 94.2% for MODA on Wildtrack and MultiviewX respectively, outperforming the state-of-the-art by 1.4% on Wildtrack and 0.5% on MultiviewX, achieving state-of-the-art performance overall for standard evaluation metrics used in multi-view pedestrian detection.

## Setup
### Data
You will need the Wildtrack & MultiviewX datasets for this repository to work.

### Docker
```
cd docker
sh build_docker.sh // Build the Docker image
```

Fill in the BASE_DIR, DATA_DIR in `docker/run_docker.sh` for the volumes to be mounted

```
sh run_docker.sh
```

Now you should be able to work inside the running Docker container.
This script will have you working inside the container so I recommend using tmux or running a slight variant of the command in `run_docker.sh` to keep the container alive even if you leave the terminal.

## Running the code
### Training & Inference
From hereon, we assume the user is in the Docker container.
BASE_DIR = /workspace
DATA_DIR = /workspace/Data
for ease of explanation.

```
python3 main.py --model BoosterSHOT --optimizer Adam --cls_thres 0.5 --depth_scales 4 --epochs 7 --augmentation True
```

### Some Additional Explanations About Arguments
`--dropcam`
- a float value between 0 and 1
- represents the ratio of frames for which at least one of the camera is "dropped"
- "dropped" cameras will have their image replaced with an all-black image to simulate camera failure
`--depth_scales`
- number of homography planes

`--world_feat`
- the type of architecture to use in the final part of the model (post-homography)

# Link to original paper
PDF

https://openaccess.thecvf.com/content/WACV2024/papers/Hwang_Booster-SHOT_Boosting_Stacked_Homography_Transformations_for_Multiview_Pedestrian_Detection_With_WACV_2024_paper.pdf

supplementary

https://openaccess.thecvf.com/content/WACV2024/supplemental/Hwang_Booster-SHOT_Boosting_Stacked_WACV_2024_supplemental.pdf

# Citation
To cite this paper, please use the bibtex below
```
@InProceedings{Hwang_2024_WACV,
    author    = {Hwang, Jinwoo and Benz, Philipp and Kim, Pete},
    title     = {Booster-SHOT: Boosting Stacked Homography Transformations for Multiview Pedestrian Detection With Attention},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {363-372}
}
```
