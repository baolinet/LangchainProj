# 在BGEM3FlagModel中使用Ollama本地BGE-M3模型

本文档介绍了如何在BGEM3FlagModel中使用本地ollama安装的bge-m3模型，提供了三种不同的解决方案。

## 前提条件

1. 确保ollama正在运行：
```bash
ollama serve
```

2. 确认bge-m3模型已安装：
```bash
ollama list
# 应该能看到 bge-m3:567m 模型
```

3. 测试模型是否正常工作：
```bash
ollama show bge-m3:567m
```

## 方案1：使用LangChain的OllamaEmbeddings（推荐）

**文件：** `BGEM3FlagModel.py`（已修改为使用兼容包装器）

**优点：**
- 简单直接
- 使用成熟的LangChain集成
- 性能良好

**代码示例：**
```python
from langchain_ollama import OllamaEmbeddings
import numpy as np

# 使用ollama中的bge-m3:567m模型
model = OllamaEmbeddings(model="bge-m3:567m")

# 生成嵌入向量
queries = ["What is BGE M3?"]
query_embeddings = []
for query in queries:
    embedding = model.embed_query(query)
    query_embeddings.append(embedding)

query_embeddings = np.array(query_embeddings)
```

## 方案2：直接使用Ollama API

**文件：** `BGEM3FlagModel_ollama_api.py`

**优点：**
- 直接控制API调用
- 不依赖额外的库
- 可以自定义请求参数

**特点：**
- 使用 `http://localhost:11434/api/embed` 端点
- 返回1024维向量
- 支持批量处理

**代码示例：**
```python
import requests
import numpy as np

def get_embedding(text, model_name="bge-m3:567m"):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": model_name, "input": text},
        timeout=30
    )
    return response.json()["embeddings"][0]
```

## 方案3：兼容BGEM3FlagModel接口的包装器（最佳选择）

**文件：** `BGEM3FlagModel_compatible.py`

**优点：**
- 完全兼容原BGEM3FlagModel接口
- 无需修改现有代码
- 支持所有原始参数
- 可以直接替换使用

**使用方法：**
```python
# 原来的代码
# from FlagEmbedding import BGEM3FlagModel
# model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# 新的代码 - 接口完全相同
from BGEM3FlagModel_compatible import OllamaBGEM3FlagModel
model = OllamaBGEM3FlagModel('bge-m3:567m', use_fp16=True)

# 其余代码完全不变
query_embeddings = model.encode(queries, batch_size=12, max_length=8192)['dense_vecs']
docs_embeddings = model.encode(docs)['dense_vecs']
similarity = query_embeddings @ docs_embeddings.T
```

## 性能对比

所有三种方案都使用相同的ollama后端，因此性能基本相同：

- **嵌入向量维度：** 1024
- **模型大小：** 566.70M参数
- **上下文长度：** 8192 tokens
- **量化：** F16

## 测试结果

使用相同的测试数据，三种方案都产生了一致的结果：

```
查询1: "What is BGE M3?"
查询2: "Defination of BM25"

文档1: "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction."
文档2: "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"

相似度矩阵:
[[0.62592124 0.34707815]
 [0.34880905 0.67823527]]
```

## 推荐使用

**对于新项目：** 使用方案1（LangChain OllamaEmbeddings）
**对于现有项目：** 使用方案3（兼容包装器），可以无缝替换原有代码

## 注意事项

1. **Sparse和ColBERT向量：** ollama的bge-m3模型目前只支持dense向量，不支持sparse和colbert向量
2. **批处理：** 当前实现是逐个处理文本，可以根据需要优化为真正的批处理
3. **错误处理：** 包含了基本的错误处理和fallback机制
4. **网络超时：** API调用设置了30秒超时

## 故障排除

1. **连接错误：** 确保ollama服务正在运行（默认端口11434）
2. **模型未找到：** 确认模型名称正确（bge-m3:567m）
3. **空向量：** 检查ollama日志，可能是模型加载问题

## 扩展功能

兼容包装器支持以下BGEM3FlagModel的参数：
- `normalize_embeddings`: 向量标准化
- `query_instruction_for_retrieval`: 查询指令
- `query_instruction_format`: 指令格式
- `use_fp16`: FP16精度（兼容性参数）
- 其他参数保持兼容性
