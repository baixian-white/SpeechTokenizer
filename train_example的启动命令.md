## 设置GPU fish版本
set -gx CUDA_VISIBLE_DEVICES 0,1 
### bash 版本为
export CUDA_VISIBLE_DEVICES=0,1

## 在启动前加环境变量（缓解碎片、提升分配成功率）fish版本
set -gx PYTORCH_CUDA_ALLOC_CONF "max_split_size_mb:64,expandable_segments:True" 
### bash 版本为
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True"
# 执行训练脚本
accelerate launch scripts/train_example.py --config config/spt_base_cfg.json

# 继续训练的脚本
accelerate launch scripts/train_example.py --config config/spt_base_cfg.json --continue_train

# tnux相关命令
tmux ls     # 列出所有tmux会话
tmux new  -s 会话名称  #创建会话
ctrl + b + d  # 退出当前会话
tmux attach -t 会话名称  # 启动会话
tmux kill-session -t <session_name>


# 启动example.py命令
python example.py \
  --speech_file samples/example_input.wav \
  --output_file example_output.wav \
  --config_path Log/spt_base/config.json \
  --ckpt_path Log/spt_base/SpeechTokenizer_best_dev.pt


# 其他命令
### 回到上一级目录
cd ..


# 在 logs/spt_base 目录下启动 TensorBoard
tensorboard --logdir=Log/spt_base --port 6007
# 查看显存情况
watch -n 1 nvidia-smi



# 将已经添加到git管理的文件剔除管理
git rm -r --cached <file_dir>

# 列出所有已添加到git管理的文件
git lst-files 
git status