## list envs
python .\loco_transformer\scripts\list_envs.py

## play
python .\loco_transformer\scripts\play.py --task <task_name> --num-envs 1 --checkpoint <checkpoint>

## train
CUDA_VISIBLE_DEVICES=0,1,2,3 
python -m torch.distributed.run --standalone --nproc_per_node=4 loco_transformer/scripts/train.py --task=<taskName> --headless --num_envs=<name>

## 显存估计
| 环境数 | 整卡显存 |
|---:|---:|
| 1 | 5070 MiB |
| 256 | 5155 MiB |
| 512 | 5333 MiB |
| 1024 | 5502 MiB |
| 2048 | 5899 MiB |
| 4096 | 6619 MiB |