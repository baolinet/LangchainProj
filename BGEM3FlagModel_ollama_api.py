# 方案2：直接使用ollama API调用bge-m3模型
import requests
import numpy as np

class OllamaBGEM3:
    def __init__(self, model_name="bge-m3:567m", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/embed"
    
    def encode(self, texts, batch_size=12, max_length=8192):
        """
        编码文本为嵌入向量
        
        Args:
            texts: 文本列表或单个文本
            batch_size: 批处理大小（这里主要用于兼容性）
            max_length: 最大长度（这里主要用于兼容性）
        
        Returns:
            dict: 包含'dense_vecs'键的字典，值为嵌入向量数组
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    self.api_url,
                    json={"model": self.model_name, "input": text},
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                # ollama的embed API返回的是embeddings数组，取第一个
                embedding = result["embeddings"][0]
                embeddings.append(embedding)
            except Exception as e:
                print(f"获取嵌入向量时出错: {e}")
                # 返回零向量作为fallback
                embeddings.append([0.0] * 1024)  # bge-m3通常是1024维
        
        return {'dense_vecs': np.array(embeddings)}
    
    def embed_query(self, text):
        """单个查询的嵌入向量"""
        result = self.encode([text])
        return result['dense_vecs'][0]

# 使用示例
if __name__ == "__main__":
    # 创建模型实例
    model = OllamaBGEM3()
    
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
    
    print("\n模型信息:")
    print(f"模型名称: {model.model_name}")
    print(f"查询嵌入向量形状: {query_embeddings.shape}")
    print(f"文档嵌入向量形状: {docs_embeddings.shape}")
