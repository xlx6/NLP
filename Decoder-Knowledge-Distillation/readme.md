# Decoder KD

## 实验设置

### 3.1 数据集
- 使用 **Alpaca** 格式指令微调数据集
- 格式：`{"instruction": ..., "input": ..., "output": ...}`
- 测试集：使用2048条数据作为测试集，模型微调后，使用测试集推理，输出格式`{"instruction": ..., "input": ..., "reference_output": ..., "generate_output":...}`


### 3.2 Teacher & Student 模型
|         |         Model         | Params |
| :-----: | :-------------------: | :----: |
| Teacher |  Qwen2.5-7B-Instruct  |  ~7B   |
| Student | Qwen2.5-0.5B-Instruct | ~0.5B  |


### 3.3 蒸馏方法
|        Methed        |            核心思路             |
| :------------------: | :-----------------------------: |
| **KL（前向，反向）** |          对齐`Logits`           |
|      **SeqKD**       | 对`Teacher`的软标签作交叉熵损失 |

### 3.3 评估方法

使用 `ChatGLM-4-Flash` 作为评分模型（因为它免费 :grin:)

包含四个评分维度（0-5）：

- **Correctness**：正确性
- **Completeness**：完整性
- **Relevance**：相关性
- **Fluency**：流畅性

#### Prompt

```python
system_prompt = """You are a professional evaluator of language model outputs. Your task is to score a model's response against a reference answer, based on the given instruction and input. Rate the response on a **0–5 scale** for each of the following:

1. **Correctness**: Is the information accurate and logical?
2. **Completeness**: Does it cover all required points?
3. **Relevance**: Is all content relevant to the task?
4. **Fluency**: Is the language natural, grammatically correct, and well-structured?

Return only the result in **strict JSON** format:

```json
{{
  "Correctness": ?,
  "Completeness": ?,
  "Relevance": ?,
  "Fluency": ?
}}
```"""
template_user = """
### Instruction:
{}
 
### Input:
{}
 
### Reference Output:
{}
 
### Generated Output:
{}
```



## 实验结果

|   Method(Loss_fn)   | Valid Data | Correctness | Completeness | Relevance | Fluency  |
| :-----------------: | :--------: | :---------: | :----------: | :-------: | :------: |
|   **Teacher(CE)**   |    2046    |   **4.0**   |   **4.0**    | **4.83**  | **4.73** |
|     Student(CE)     |    2045    |    3.52     |     3.53     |   4.44    |   4.38   |
| Student(Forward_KL) |    2046    |    3.53     |     3.54     |   4.46    |   4.37   |
| Student(Reverse_KL) |    2046    |    3.53     |     3.55     |   4.46    |   4.38   |
| **Student(JS_KL)**  |    2045    |  **3.55**   |   **3.57**   | **4.49**  | **4.4**  |
|   Student(SeqKD)    |    2046    |    3.51     |     3.55     |   4.43    |   4.38   |

*注：测试集中，有两条数据被判为违规数据，其余无效数据均为评测模型未按照要求响应*

结果表明，教师模型表现最佳，所有指标均优于学生模型。

使用 `KL` 蒸馏的模型均在 `CE` 的基础上带来了小幅度提升（$\alpha=0.5$)，其中 `JS_KL` （前向与反向 `KL` 对称组合）在四项指标中全面领先，说明双向 `KL` 能更好拟合教师的分布。

`SeqKD`  方法通过最小化生成序列与教师序列之间的交叉熵来进行蒸馏，理论上能更好地捕捉教师的语言风格与输出分布。然而在实际评估中，其性能与标准 CE 相当。