set -e

cd ./submodules/GEN3C

mkdir -p outputs

for cam in 0000 0001 0002 0003 0004 0005 0006 0007; do
    mkdir -p outputs/f121_statesize/${cam}
    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/gen3c_dynamic.py \
        --checkpoint_dir checkpoints \
        --input_image_path ./gen3c_testdata/f121-statesize/${cam}.pt \
        --trajectory left \
        --camera_rotation no_rotation \
        --movement_distance 1.5 \
        --video_save_name dynamic_video \
        --video_save_folder ./outputs/f121_statesize/${cam} \
        --num_video_frames 121 \
        --height 720 \
        --width 1280 \
        --fps 10 \
        --guidance 1 \
        --offload_diffusion_transformer \
        --offload_tokenizer \
        --offload_text_encoder_model \
        --offload_prompt_upsampler \
        --disable_prompt_encoder \
        --offload_guardrail_models &> ./outputs/gen3c_log_f121_statesize_${cam}.txt
done

for cam in 0000 0001 0002 0003 0004 0005 0006 0007; do
    mkdir -p outputs/f241_statesize/${cam}
    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/gen3c_dynamic.py \
        --checkpoint_dir checkpoints \
        --input_image_path ./gen3c_testdata/f241_statesize/${cam}.pt \
        --trajectory left \
        --camera_rotation no_rotation \
        --movement_distance 1.5 \
        --video_save_name dynamic_video \
        --video_save_folder ./outputs/f241_statesize/${cam} \
        --num_video_frames 241 \
        --height 720 \
        --width 1280 \
        --fps 10 \
        --guidance 1 \
        --offload_diffusion_transformer \
        --offload_tokenizer \
        --offload_text_encoder_model \
        --offload_prompt_upsampler \
        --disable_prompt_encoder \
        --offload_guardrail_models &> ./outputs/gen3c_log_f241_statesize_${cam}.txt
done