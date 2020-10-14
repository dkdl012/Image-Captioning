id='Transformer_mrc_frc'
id_rl="Transformer_mrc_frc_rl"
model="log/log_$id_rl/model-best.pth"
infos_path="log/log_$id_rl/infos_$id-best.pkl"
python eval.py \
    --id $id \
    --model $model \
    --infos_path $infos_path \
    --num_cnn 2 \
    --use_mrc_feat 1 \
    --add_self 1 \
    --frc_first 0 \
    --dump_images 0 \
    --dump_json 1 \
    --num_images -1 \
    --language_eval 1 \
    --beam_size 2 \
    --batch_size 100 \
    --split test