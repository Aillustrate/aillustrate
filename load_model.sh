MODEL_ID=$1;
MODEL_NAME=$2;
wget https://civitai.com/api/download/models/$MODEL_ID --content-disposition
wget https://raw.githubusercontent.com/huggingface/diffusers/v0.20.0/scripts/convert_original_stable_diffusion_to_diffusers.py
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path $MODEL_NAME\.safetensors --dump_path $MODEL_NAME/ --from_safetensors