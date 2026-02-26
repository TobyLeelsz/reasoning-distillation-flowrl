sudo docker run -d --name verl --gpus all \
  --network host \
  --ipc=host --shm-size=64g \
  -v "$HOME/shangzhe/FlowRL:/workspace" \
  -w /workspace \
  tobyleelsz/verl_flowrl:latest \
  sleep infinity

sudo docker exec -it verl bash