#模型下载
# from modelscope import snapshot_download
# model_dir = snapshot_download('ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3', cache_dir="./model/")

#gguf模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3-gguf', cache_dir="./model/")