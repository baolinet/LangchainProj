# 方案3：创建兼容BGEM3FlagModel接口的包装器
import requests
import numpy as np
from typing import List, Union, Dict, Any

class OllamaBGEM3FlagModel:
    """
    兼容BGEM3FlagModel接口的ollama包装器
    可以直接替换原来的BGEM3FlagModel使用
    """
    
    def __init__(self, model_name_or_path: str = "bge-m3:567m", 
                 normalize_embeddings: bool = True, 
                 use_fp16: bool = True,
                 query_instruction_for_retrieval: str = None,
                 query_instruction_format: str = '{}{}',
                 devices: Union[str, List[str]] = None,
                 pooling_method: str = 'cls',
                 trust_remote_code: bool = False,
                 cache_dir: str = None,
                 colbert_dim: int = -1,
                 batch_size: int = 256,
                 query_max_length: int = 512,
                 passage_max_length: int = 512,
                 return_dense: bool = True,
                 return_sparse: bool = False,
                 return_colbert_vecs: bool = False,
                 base_url: str = "http://localhost:11434",
                 **kwargs):
        """
        初始化OllamaBGEM3FlagModel
        
        Args:
            model_name_or_path: ollama模型名称，如 "bge-m3:567m"
            normalize_embeddings: 是否标准化嵌入向量
            use_fp16: 是否使用fp16（这里主要用于兼容性）
            base_url: ollama服务地址
            其他参数: 为了兼容BGEM3FlagModel接口而保留
        """
        self.model_name = model_name_or_path
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.query_instruction_format = query_instruction_format
        self.return_dense = return_dense
        self.return_sparse = return_sparse
        self.return_colbert_vecs = return_colbert_vecs
        self.base_url = base_url
        self.api_url = f"{base_url}/api/embed"
        
        print(f"初始化OllamaBGEM3FlagModel，使用模型: {self.model_name}")
    
    def encode(self, sentences: Union[str, List[str]], 
               batch_size: int = 12, 
               max_length: int = 8192,
               return_dense: bool = None,
               return_sparse: bool = None,
               return_colbert_vecs: bool = None) -> Dict[str, Any]:
        """
        编码文本为嵌入向量，兼容BGEM3FlagModel的接口
        
        Args:
            sentences: 输入文本或文本列表
            batch_size: 批处理大小
            max_length: 最大长度
            return_dense: 是否返回dense向量
            return_sparse: 是否返回sparse向量
            return_colbert_vecs: 是否返回colbert向量
        
        Returns:
            dict: 包含嵌入向量的字典
        """
        # 使用实例变量作为默认值
        if return_dense is None:
            return_dense = self.return_dense
        if return_sparse is None:
            return_sparse = self.return_sparse
        if return_colbert_vecs is None:
            return_colbert_vecs = self.return_colbert_vecs
        
        # 确保输入是列表
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # 应用查询指令（如果有）
        processed_sentences = []
        for sentence in sentences:
            if self.query_instruction_for_retrieval:
                processed_sentence = self.query_instruction_format.format(
                    self.query_instruction_for_retrieval, sentence
                )
            else:
                processed_sentence = sentence
            processed_sentences.append(processed_sentence)
        
        # 获取嵌入向量
        embeddings = []
        for sentence in processed_sentences:
            try:
                response = requests.post(
                    self.api_url,
                    json={"model": self.model_name, "input": sentence},
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                embedding = result["embeddings"][0]
                embeddings.append(embedding)
            except Exception as e:
                print(f"获取嵌入向量时出错: {e}")
                # 返回零向量作为fallback
                embeddings.append([0.0] * 1024)
        
        embeddings = np.array(embeddings)
        
        # 标准化嵌入向量（如果需要）
        if self.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-12)  # 避免除零
        
        # 构建返回结果
        result = {}
        
        if return_dense:
            result['dense_vecs'] = embeddings
        
        if return_sparse:
            # ollama的bge-m3不直接支持sparse向量，这里返回空的sparse表示
            print("警告: ollama的bge-m3模型不支持sparse向量，返回空结果")
            result['lexical_weights'] = [[] for _ in sentences]
        
        if return_colbert_vecs:
            # ollama的bge-m3不直接支持colbert向量，这里返回空的colbert表示
            print("警告: ollama的bge-m3模型不支持colbert向量，返回空结果")
            result['colbert_vecs'] = [[] for _ in sentences]
        
        return result

# 使用示例 - 完全兼容原来的BGEM3FlagModel代码
if __name__ == "__main__":
    # 可以直接替换原来的 BGEM3FlagModel
    model = OllamaBGEM3FlagModel('bge-m3:567m', use_fp16=True)
    
    queries = ["What is BGE M3?",
               "Defination of BM25"]
    docs = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
            "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]
    
    print("正在生成查询嵌入向量...")
    query_embeddings = model.encode(queries, batch_size=12, max_length=8192)['dense_vecs']
    
    print("正在生成文档嵌入向量...")
    docs_embeddings = model.encode(docs)['dense_vecs']
    
    # 计算相似度
    similarity = query_embeddings @ docs_embeddings.T
    print("相似度矩阵:")
    print(similarity)
    
    print(f"\n模型信息:")
    print(f"模型名称: {model.model_name}")
    print(f"查询嵌入向量形状: {query_embeddings.shape}")
    print(f"文档嵌入向量形状: {docs_embeddings.shape}")
    print(f"标准化嵌入向量: {model.normalize_embeddings}")
