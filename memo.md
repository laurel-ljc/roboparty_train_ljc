## list envs

```bash
python .\loco_transformer\scripts\list_envs.py
```

## play

```bash
python .\loco_transformer\scripts\play.py --task <task_name> --num-envs 1 --checkpoint <checkpoint>
```

## train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=<num_of_cards> loco_transformer/scripts/train.py --task=<taskName> --headless --num_envs=<num> --finetune-path=<check_point_path>
```

## 显存估计

### RPO-Loco-Transformer

| 环境数  | 整卡显存     |
| ----:| --------:|
| 1    | 5070 MiB |
| 256  | 5155 MiB |
| 512  | 5333 MiB |
| 1024 | 5502 MiB |
| 2048 | 5899 MiB |
| 4096 | 6619 MiB |

### RPO-Loco-Transformer-History-Rough

| 环境数  | 整卡峰值显存                                           |
| ---- | ------------------------------------------------ |
| 1    | 6819 MiB（6.66 GiB）                               |
| 128  | 7607 MiB（7.43 GiB）                               |
| 256  | 9311 MiB（9.09 GiB）                               |
| 1024 | 12093 MiB(10.05 GiB) |

## 进程确认

```bash
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sort -u); do
    echo -n "$pid: "
    tr '\0' ' ' < /proc/$pid/cmdline
    echo
done
```

## 清理进程

```bash
pkill -9 -f '/envs/roboparty/'
```
