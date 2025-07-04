export MODELSCOPE_CACHE='/root/autodl-fs/'
# ./hfd.sh Qwen/Qwen3-14B --local-dir Qwen3-14B
# ./hfd.sh Qwen/Qwen3-0.6B --local-dir Qwen3-0.6B
# ./hfd.sh microsoft/phi-4 --local-dir phi-4
# ./hfd.sh microsoft/Phi-4-mini-instruct --local-dir Phi-4-mini-instruct
# ./hfd.sh mistralai/Mistral-7B-Instruct-v0.3 --local-dir Mistral-7B-Instruct-v0.3
# ./hfd.sh google/gemma-3n-E4B-it --local-dir gemma-3n-E4B-it
# ./hfd.sh tencent/Hunyuan-A13B-Instruct --local-dir Hunyuan-A13B-Instruct
modelscope download Qwen/Qwen3-8B
modelscope download Qwen/Qwen3-0.6B
modelscope download Qwen/Qwen2.5-0.5B
modelscope download Qwen/Qwen2.5-7B
modelscope download microsoft/phi-4
modelscope download microsoft/Phi-4-mini-instruct
modelscope download mistralai/Mistral-7B-Instruct-v0.3
modelscope download google/gemma-3n-E4B-it