CONFIG="config/spt_base_cfg.json"
# 音频目录（LibriSpeech 的 train-clean-100 到这里）
AUDIO_DIR="$PWD/data/SpeechPretrain/LibriSpeech/train-clean-100"

# 语义表征输出目录（脚本会自动创建子目录并写入 .hubert.npy）
REP_DIR="$PWD/data/SpeechPretrain/hubert_rep/LibriSpeech"

EXTS="flac"
SPLIT_SEED=0
VALID_SET_SIZE=1500



CUDA_VISIBLE_DEVICES=0 python scripts/hubert_rep_extract.py\
    --config ${CONFIG}\
    --audio_dir ${AUDIO_DIR}\
    --rep_dir ${REP_DIR}\
    --exts ${EXTS}\
    --split_seed ${SPLIT_SEED}\
    --valid_set_size ${VALID_SET_SIZE}