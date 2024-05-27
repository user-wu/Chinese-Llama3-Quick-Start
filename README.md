# Chinese-llama3-fastdemo

# 模型简介

本项目基于Meta最新发布的新一代开源大模型Llama-3开发，是Chinese-LLaMA-Alpaca开源大模型相关系列项目的第三期。项目开源的中文Llama-3基座模型和中文Llama-3-Instruct指令精调大模型在原版Llama-3的基础上使用了大规模中文数据进行增量预训练，并且使用精选指令数据进行精调，进一步提升了中文基础语义和指令理解能力，相比二代相关模型获得了显著性能提升。

## 内容导引
| 章节                                  | 描述                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| [💁🏻‍♂️模型简介](#模型简介) | 简要介绍本项目相关模型的技术特点 |
| [⏬模型下载](#模型下载)        | 中文Llama-3大模型下载地址 |
| [💻推理与部署](#推理与部署) | 介绍了如何对模型进行量化并使用个人电脑部署并体验大模型 |
| [📝训练与精调](#训练与精调) | 介绍了如何训练和精调中文Llama-3大模型 |
| [❓常见问题](#常见问题) | 一些常见问题的回复 |

# 模型下载

### 下载地址

| 模型名称                  |                    完整版                    |                    LoRA版                    |                    GGUF版                    |
| :------------------------ | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **Llama-3-Chinese-8B-Instruct-v2**<br/>(指令模型) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-v2)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-v2-lora)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2-lora)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2-lora) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-v2-gguf)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2-gguf) |
| **Llama-3-Chinese-8B-Instruct**<br/>(指令模型) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-lora)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-lora)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-lora) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-gguf)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-gguf) |
| **Llama-3-Chinese-8B**<br/>(基座模型) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-lora)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-lora)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-lora) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-gguf)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-gguf) |

模型类型说明：

- **完整模型**：可直接用于训练和推理，无需其他合并步骤
- **LoRA模型**：需要与基模型合并并才能转为完整版模型，合并方法：[**💻 模型合并步骤**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/model_conversion_zh)
  - v1基模型：原版[Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
  - v2基模型：原版[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- **GGUF模型**：[llama.cpp](https://github.com/ggerganov/llama.cpp)推出的量化格式，适配ollama等常见推理工具，推荐只需要做推理部署的用户下载；模型名后缀为`-im`表示使用了importance matrix进行量化，通常具有更低的PPL，建议使用（用法与常规版相同）
> [!NOTE]
> 若无法访问HF，可考虑一些镜像站点（如[hf-mirror.com](hf-mirror.com)），具体方法请自行查找解决。

# 推理与部署

# 训练与精调

# 常见问题
