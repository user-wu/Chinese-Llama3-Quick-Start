# Chinese-llama3-fastdemo
## 内容导引
| 章节                                  | 描述                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| [模型简介](#一、模型简介) | 简要介绍本项目相关模型的技术特点 |
| [模型下载](#二、模型下载)        | 中文Llama-3大模型下载地址 |
| [推理与部署](#三、推理与部署) | 介绍了如何对模型进行量化并使用个人电脑部署并体验大模型 |
| [训练与精调](#四、训练与精调) | 介绍了如何训练和精调中文Llama-3大模型 |
| [免责声明](#五、免责声明) | 相关免责声明 |

# 一、模型简介

本项目基于Meta最新发布的新一代开源大模型Llama-3开发，是Chinese-LLaMA-Alpaca开源大模型相关系列项目的第三期。项目开源的中文Llama-3基座模型和中文Llama-3-Instruct指令精调大模型在原版Llama-3的基础上使用了大规模中文数据进行增量预训练，并且使用精选指令数据进行精调，进一步提升了中文基础语义和指令理解能力，相比二代相关模型获得了显著性能提升。

# 二、模型下载
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

# 三、推理与部署

## 3.1 使用transformers进行推理
我们提供了命令行方式使用原生🤗transformers进行推理。下面以加载Llama-3-Chinese-Instruct模型为例说明启动方式。
下载完整版权重之后，按以下命令启动脚本。
```
python scripts/inference/inference_hf.py \
    --base_model path_to_llama3_chinese_instruct_hf_dir \
    --with_prompt \
    --interactive
```

#### 使用vLLM进行推理加速
可以使用vLLM作为LLM后端进行推理，需要额外安装vLLM库。
```
pip install vllm
```
只需在原本的命令行上添加`--use_vllm`参数:
```
python scripts/inference/inference_hf.py \
    --base_model path_to_llama3_chinese_instruct_hf_dir \
    --with_prompt \
    --interactive \
    --use_vllm
```
#### 参数说明
* `--base_model {base_model}` ：存放HF格式的Llama-3-Chinese-Instruct模型权重和配置文件的目录。也可使用🤗Model Hub模型调用名称
* `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与--base_model相同
* `--with_prompt`：是否将输入与prompt模版进行合并。如果加载Llama-3-Chinese-instruct模型，请务必启用此选项！
* `--interactive`：以交互方式启动，以便进行多次单轮问答（此处不是llama.cpp中的上下文对话）
* `--data_file {file_name}`：非交互方式启动下，按行读取file_name中的的内容进行预测
* `--predictions_file {file_name}`：非交互式方式下，将预测的结果以json格式写入file_name
* `--only_cpu`：仅使用CPU进行推理
* `--gpus {gpu_ids}`：指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如0,1,2
* `--load_in_8bit`或`--load_in_4bit`：使用8bit或4bit方式加载模型，降低显存占用，推荐使用--load_in_4bit
* `--use_vllm`：使用vLLM作为LLM后端进行推理
* `--use_flash_attention_2`: 使用Flash-Attention 2加速推理，如果不指定该参数，代码默认SDPA加速。
该脚本仅为方便快速体验用，并未对推理速度做优化。

## 3.2使用llama.cpp量化部署
以[llama.cpp](https://github.com/ggerganov/llama.cpp)工具为例，介绍模型量化并在本地部署的详细步骤。Windows则可能需要cmake等编译工具的安装。本地快速部署体验推荐使用经过指令精调的Llama-3-Chinese-Instruct模型，使用6-bit或者8-bit模型效果更佳。 运行前请确保：
* 1.系统应有make（MacOS/Linux自带）或cmake（Windows需自行安装）编译工具
* 2.建议使用Python 3.10以上编译和运行该工具
#### Step 1: 克隆和编译llama.cpp
#### llama.cpp在2024年4月30日对Llama-3 pre-tokenizer做出重大改动，务必拉取最新代码进行编译！
* 1.拉取最新版`llama.cpp`仓库代码
```
git clone https://github.com/ggerganov/llama.cpp
```
* 2.对`llama.cpp`项目进行编译，生成`./main（用于推理）`和`./quantize`（用于量化）二进制文件。
```
make
```
Windows/Linux用户如需启用GPU推理，则推荐与[BLAS（或cuBLAS如果有GPU）一起编译](https://github.com/ggerganov/llama.cpp#blas-build)，可以提高prompt处理速度。以下是和cuBLAS一起编译的命令，适用于NVIDIA相关GPU。参考：[llama.cpp#blas-build](https://github.com/ggerganov/llama.cpp#blas-build)
```
make LLAMA_CUDA=1
```
macOS用户无需额外操作，`llama.cpp`已对ARM NEON做优化，并且已自动启用BLAS。M系列芯片推荐使用Metal启用GPU推理，显著提升速度。只需将编译命令改为：LLAMA_METAL=1 make，参考[llama.cpp#metal-build](https://github.com/ggerganov/llama.cpp#metal-build)
```
LLAMA_METAL=1 make
```
#### Step 2: 生成量化版本模型
也可直接下载已量化好的GGUF模型：[下载地址](#下载地址)

目前`llama.cpp`已支持`.safetensors`文件以及`Hugging Face`格式`.bin`转换为FP16的`GGUF`格式。
$ python convert-hf-to-gguf.py llama-3-chinese-8b-instruct
$ ./quantize ggml-model-f16.gguf ggml-model-q4_0.gguf q4_0

#### Step 3: 加载并启动模型
由于本项目推出的Llama-3-Chinese-Instruct使用了原版Llama-3-Instruct的指令模板，请首先将本项目的`scripts/llama_cpp/chat.sh`拷贝至`llama.cpp`的根目录。`chat.sh`文件的内容如下所示，内部嵌套了聊天模板和一些默认参数，可根据实际情况进行修改。
* GPU推理：cuBLAS/Metal编译需要指定offload层数，在./main中指定例如-ngl 40表示offload 40层模型参数到GPU
* （新）启用FlashAttention：命令行中添加-fa即可启用，可加速推理（因计算设备而异）
```
FIRST_INSTRUCTION=$2
SYSTEM_PROMPT="You are a helpful assistant. 你是一个乐于助人的助手。"

./main -m $1 --color -i \
-c 0 -t 6 --temp 0.2 --repeat_penalty 1.1 -ngl 999 \
-r '<|eot_id|>' \
--in-prefix '<|start_header_id|>user<|end_header_id|>\n\n' \
--in-suffix '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' \
-p "<|start_header_id|>system<|end_header_id|>\n\n$SYSTEM_PROMPT<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n$FIRST_INSTRUCTION<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
```
使用以下命令启动聊天。
```
chmod +x chat.sh
./chat.sh ggml-model-q4_0.gguf 你好
```
在提示符 > 之后输入你的prompt，cmd/ctrl+c中断输出，多行信息以\作为行尾。如需查看帮助和参数说明，请执行./main -h命令。
更详细的官方说明请参考：[https://github.com/ggerganov/llama.cpp/tree/master/examples/main](https://github.com/ggerganov/llama.cpp/tree/master/examples/main)

## 3.3使用ollama部署
[Ollama](https://ollama.com/)是一个多平台（macOS, Windows, Linux）的大模型聊天程序，能够加载GGUF格式（llama.cpp）的模型。接下来将简要介绍使用方法。其余用途请自行尝试和查阅官方手册进行了解。

#### Step 1: 下载对应平台的应用程序
进入官方页面下载对应平台的软件：https://ollama.com/download
* ⚠️ 请务必使用v0.1.33以上版本，否则会出现无限生成的问题。
  ![image](https://github.com/user-wu/Chinese-llama3-fastdemo/assets/67259115/491bdcc2-98e3-4aad-817c-520e667ab794)
#### Step 2: 安装Ollama
* macOS：下载完毕之后直接拖入“应用程序”
* Windows preview：下载运行exe文件
* Linux：执行以下命令
```
curl -fsSL https://ollama.com/install.sh | sh
```
其余平台请参考：[https://github.com/ollama/ollama?tab=readme-ov-file#ollama](https://github.com/ollama/ollama?tab=readme-ov-file#ollama)
#### Step 3：创建Modelfile文件
在文本编辑器中编写`Modelfile`文件，其内容如下：
```
FROM /your-path-to-ggml/ggml-model-q8_0.gguf
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
SYSTEM """You are a helpful assistant. 你是一个乐于助人的助手。"""
PARAMETER temperature 0.2
PARAMETER num_keep 24
PARAMETER stop <|start_header_id|>
PARAMETER stop <|end_header_id|>
PARAMETER stop <|eot_id|>
```
其中：

* `FROM`字段指向GGUF文件的路径，由于是聊天交互，这里使用的是Instruct模型
* `TEMPLATE`字段定义了Llama-3-Instruct的指令模板格式
* `SYSTEM`字段定义了系统指令（目前设置为空）
* `PARAMETER`字段定义了一些超参数，详细列表参见：[https://github.com/ollama/ollama/blob/main/docs/modelfile.md](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)

#### Step 4：创建模型实例
命令行中运行以下命令，创建一个名为`llama3-zh-inst`（名字可自定义）的模型实例，加载`Modelfile`配置：
```
ollama create llama3-zh-inst -f Modelfile
```
创建过程输出日志如下：
```
transferring model data
creating model layer
creating template layer
creating system layer
creating parameters layer
creating config layer
using already created layer sha256:f2a44c6358e8e0a60337f8a1b31f55f457558eeefd4f344272e44b0e73a86a32
using already created layer sha256:8ab4849b038cf0abc5b1c9b8ee1443dca6b93a045c2272180d985126eb40bf6f
writing layer sha256:b821abf159071cfc90f0941b5ca7ef721f229cfcfadcf95b5c58d0ceb3e773c7
writing layer sha256:dc4ec177268acc3382fc6c3a395e577bf13e9e0340dd313a75f62df95c48bc1d
writing manifest
success
```
输出`success`后，即表示完成创建。
#### Step 5：开始聊天
输入以下命令进入聊天程序
```
ollama run llama3-zh-inst
```
在>>>后输入用户指令；输入/bye结束聊天。

关于ollama的其他用法，请参考官方文档：https://github.com/ollama/ollama?tab=readme-ov-file#cli-reference

# 四、训练与精调

### 训练步骤
[训练脚本](./scripts/training/run_clm_pt_with_peft.py)

进入项目的`scripts/training`目录，运行`bash run_pt.sh`进行指令精调，默认使用单卡。运行前用户应先修改脚本并指定相关参数，脚本中的相关参数值仅供调试参考。`run_pt.sh`的内容如下：
```
########参数设置########
lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=path/to/hf/meta-llama-3-8b/dir
dataset_dir=path/to/pt/data/dir
data_cache=temp_data_cache_dir
per_device_train_batch_size=1
gradient_accumulation_steps=8
block_size=1024
output_dir=output_dir

torchrun --nnodes 1 --nproc_per_node 1 run_clm_pt_with_peft.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --seed $RANDOM \
    --bf16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --torch_dtype bfloat16 \
    --load_in_kbits 16 \
    --ddp_find_unused_parameters False \
```

部分参数的解释如下：

* `--dataset_dir`: 预训练数据的目录，可包含多个以txt结尾的纯文本文件
* `--data_cache_dir`: 指定一个存放数据缓存文件的目录
* `--use_flash_attention_2`: 启用FlashAttention-2加速训练
* `--load_in_kbits`: 可选择参数为16/8/4，即使用fp16或8bit/4bit量化进行模型训练，默认bf16训练。
* `--modules_to_save`：需要额外训练的模块，注意这部分是全量精调；资源受限的情况下请设置为None（效果也会受到一些影响）
这里列出的其他训练相关超参数，尤其是学习率以及和total batch size大小相关参数仅供参考。请在实际使用时根据数据情况以及硬件条件进行配置。

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

# 五、免责声明
本项目基于由Meta发布的Llama-3模型进行开发，供学习使用。使用过程中请严格遵守Llama-3的[开源许可协议](https://github.com/meta-llama/llama3/blob/main/LICENSE)。如果涉及使用第三方代码，请务必遵从相关的开源许可协议。模型生成的内容可能会因为计算方法、随机因素以及量化精度损失等影响其准确性，因此，本项目不对模型输出的准确性提供任何保证，也不会对任何因使用相关资源和输出结果产生的损失承担责任。如果将本项目的相关模型用于商业用途，开发者应遵守当地的法律法规，确保模型输出内容的合规性，本项目不对任何由此衍生的产品或服务承担责任。

