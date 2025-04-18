model_dir=${1:-"/data/home/0006179156/scow/data_para_model/caption_pllava/pllava7b"}
weight_dir=${2:-"/data/home/0006179156/scow/data_para_model/caption_pllava/pllava7b"}
num_frames=4
lora_alpha=4

echo Running DEMO from model_dir: ${model_dir}
echo Running DEMO from weights_dir: ${weight_dir}
echo Running DEMO On Devices: ${CUDA_VISIBLE_DEVICES}


# # 34B Need to Use dispatch for this large.
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m tasks.eval.demo.pllava_demo \
#     --pretrained_model_name_or_path ${model_dir} \
#     --num_frames ${num_frames} \
#     --use_lora \
#     --weight_dir ${weight_dir} \
#     --lora_alpha ${lora_alpha} \
#     --conv_mode eval_vcg_llava_next \
#     --use_multi_gpus \


# 7B and 13B, There are problem if Model was split around A100 40G... Probably because some unkown bug in accelerate dispatch
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"} python -m tasks.eval.demo.pllava_demo \
    --pretrained_model_name_or_path ${model_dir} \
    --num_frames ${num_frames} \
    --use_lora \
    --weight_dir ${weight_dir} \
    --lora_alpha ${lora_alpha} \
    --conv_mode plain


