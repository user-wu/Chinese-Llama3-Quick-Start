# Chinese-llama3-fastdemo
## 内容导引
| 章节                                  | 描述                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| [💁🏻‍♂️模型简介](#模型简介) | 简要介绍本项目相关模型的技术特点 |
| [⏬模型下载](#模型下载)        | 中文Llama-3大模型下载地址 |
| [💻推理与部署](#推理与部署) | 介绍了如何对模型进行量化并使用个人电脑部署并体验大模型 |
| [📝训练与精调](#训练与精调) | 介绍了如何训练和精调中文Llama-3大模型 |
| [❓常见问题](#常见问题) | 一些常见问题的回复 |

# 模型简介

本项目基于Meta最新发布的新一代开源大模型Llama-3开发，是Chinese-LLaMA-Alpaca开源大模型相关系列项目的第三期。项目开源的中文Llama-3基座模型和中文Llama-3-Instruct指令精调大模型在原版Llama-3的基础上使用了大规模中文数据进行增量预训练，并且使用精选指令数据进行精调，进一步提升了中文基础语义和指令理解能力，相比二代相关模型获得了显著性能提升。

# 模型下载
### 模型选择指引

以下是本项目的模型对比以及建议使用场景。**如需聊天交互，请选择Instruct版。**

| 对比项                | Llama-3-Chinese-8B             | Llama-3-Chinese-8B-Instruct |
| :-------------------- | :----------------------------------------------------: | :----------------------------------------------------------: |
| 模型类型 | 基座模型 | 指令/Chat模型（类ChatGPT） |
| 模型大小 | 8B | 8B |
| 训练类型     | Causal-LM (CLM)           | 指令精调                                                     |
| 训练方式 | LoRA + 全量emb/lm-head | LoRA + 全量emb/lm-head |
| 初始化模型 | [原版Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | v1: Llama-3-Chinese-8B<br/>v2: [原版Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| 训练语料 | 无标注通用语料（约120GB） | 有标注指令数据（约500万条） |
| 词表大小 | 原版词表（128,256） | 原版词表（128,256） |
| 支持上下文长度 | 8K | 8K |
| 输入模板              | 不需要                                                 | 需要套用Llama-3-Instruct模板 |
| 适用场景            | 文本续写：给定上文，让模型生成下文            | 指令理解：问答、写作、聊天、交互等 |

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

| [🤗transformers](https://github.com/huggingface/transformers) | 原生transformers推理接口     |  ✅   |  ✅   |  ✅   |  ✅   |  ❌   |  ✅  | [[link]](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/inference_with_transformers_zh) |

| [Ollama](https://github.com/ollama/ollama) | 本地运行大模型推理 | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | [[link]](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/ollama_zh) |

# 训练与精调

### 手动训练与精调

- 使用无标注数据进行预训练：[📖预训练脚本Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/pt_scripts_zh)
- 使用有标注数据进行指令精调：[📖指令精调脚本Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/sft_scripts_zh)

### 指令模板

本项目Llama-3-Chinese-Instruct沿用原版Llama-3-Instruct的指令模板。以下是一组对话示例：

> <|begin_of_text|><|start_header_id|>system<|end_header_id|>
>
> You are a helpful assistant. 你是一个乐于助人的助手。<|eot_id|><|start_header_id|>user<|end_header_id|>
>
> 你好<|eot_id|><|start_header_id|>assistant<|end_header_id|>
>
> 你好！有什么可以帮助你的吗？<|eot_id|>

### 指令数据

以下是本项目开源的部分指令数据。详情请查看：[📚 指令数据](./data)

# 免责声明
本项目基于由Meta发布的Llama-3模型进行开发，供学习使用。使用过程中请严格遵守Llama-3的[开源许可协议](https://github.com/meta-llama/llama3/blob/main/LICENSE)。如果涉及使用第三方代码，请务必遵从相关的开源许可协议。模型生成的内容可能会因为计算方法、随机因素以及量化精度损失等影响其准确性，因此，本项目不对模型输出的准确性提供任何保证，也不会对任何因使用相关资源和输出结果产生的损失承担责任。如果将本项目的相关模型用于商业用途，开发者应遵守当地的法律法规，确保模型输出内容的合规性，本项目不对任何由此衍生的产品或服务承担责任。

