#!/usr/bin/env bash

# Some distro requires that the absolute path is given when invoking lspci
# e.g. /sbin/lspci if the user is not root.
echo 'Looking for GPUs (ETA: 10 seconds)'
gpu=$(lspci | grep -i '.* vga .* nvidia .*')
shopt -s nocasematch

if [[ $gpu == *' nvidia '* ]]; then
  echo GPU found
  docker run -it --rm \
    --privileged=true \
    --mount "type=bind,src=$(pwd),dst=/tmp/" \
    --mount "type=bind,src=/home/storage/maperezc/CloudMethaneSat/data,dst=/tmp/data" \
    --workdir /tmp/ \
    --gpus '"device=1"' \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name methane \
    -p 8889:8889 \
    methane bash
else
  echo "No GPU found. Running in CPU-only mode."
  docker run -it --rm \
    --privileged=true \
    --mount "type=bind,src=$(pwd),dst=/tmp/" \
    --mount "type=bind,src=/home/storage/maperezc/CloudMethaneSat/data,dst=/tmp/data" \
    --workdir /tmp/ \
    --name methane \
    -p 8889:8889 \
    methane bash
fi
