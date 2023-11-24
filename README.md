# BoosterSHOT
Repository containing code for BoosterSHOT

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
