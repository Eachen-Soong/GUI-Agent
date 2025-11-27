CUDA_VISIBLE_DEVICES=0 vllm serve ./QwenVL-4B \
    --served-model-name Qwen/Qwen3-VL-4B-Instruct \
    --max_model_len 4096 \
    --port 6666 \
    --gpu-memory-utilization 0.7 \
    # --api-key 114514 \

