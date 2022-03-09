docker run \
    -it --rm \
    --gpus all \
    --shm-size=32G \
    --publish PORT:PORT \
    --volume BASE_DIR:/workspace/ \
    --volume DATA_DIR:/workspace/Data/ \
    boostershot/research