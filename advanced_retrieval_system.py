# 高级文档检索系统
from BGEM3FlagModel_compatible import OllamaBGEM3FlagModel
import numpy as np
from typing import List, Tuple, Dict
import json
from datetime import datetime

class AdvancedRetrievalSystem:
    def __init__(self, model_name: str = "bge-m3:567m"):
        """初始化高级检索系统"""
        self.model = OllamaBGEM3FlagModel(model_name, use_fp16=True)
        self.documents = []
        self.doc_embeddings = None
        self.doc_metadata = []
        
    def add_documents(self, docs: List[str], metadata: List[Dict] = None):
        """添加文档到检索系统"""
        self.documents.extend(docs)
        
        if metadata is None:
            metadata = [{"id": len(self.documents) + i, "added_at": datetime.now().isoformat()} 
                       for i in range(len(docs))]
        self.doc_metadata.extend(metadata)
        
        # 重新计算所有文档的嵌入向量
        print(f"正在为 {len(self.documents)} 个文档生成嵌入向量...")
        self.doc_embeddings = self.model.encode(self.documents)['dense_vecs']
        print(f"嵌入向量形状: {self.doc_embeddings.shape}")
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Tuple[int, float, str, Dict]]:
        """
        搜索最相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            threshold: 相似度阈值
            
        Returns:
            List of (index, similarity, document, metadata)
        """
        if self.doc_embeddings is None:
            raise ValueError("请先添加文档")
        
        # 生成查询嵌入向量
        query_embedding = self.model.encode([query])['dense_vecs']
        
        # 计算相似度
        similarities = query_embedding @ self.doc_embeddings.T
        similarities = similarities[0]
        
        # 获取top_k个最相似的文档
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                results.append((
                    idx, 
                    similarity, 
                    self.documents[idx], 
                    self.doc_metadata[idx]
                ))
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 3) -> Dict[str, List[Tuple]]:
        """批量搜索多个查询"""
        results = {}
        for query in queries:
            results[query] = self.search(query, top_k)
        return results
    
    def get_statistics(self) -> Dict:
        """获取系统统计信息"""
        if self.doc_embeddings is None:
            return {"documents": 0, "embedding_dim": 0}
        
        return {
            "documents": len(self.documents),
            "embedding_dim": self.doc_embeddings.shape[1],
            "model": self.model.model_name,
            "avg_doc_length": np.mean([len(doc.split()) for doc in self.documents])
        }

def create_comprehensive_knowledge_base():
    """创建综合知识库"""
    
    # 技术文档集合
    tech_docs = [
        # AI/ML 基础
        "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans.",
        "Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
        "Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns.",
        "Neural networks are computing systems inspired by biological neural networks that constitute animal brains, consisting of interconnected nodes or neurons.",
        "Supervised learning uses labeled training data to learn a mapping function from input variables to output variables.",
        "Unsupervised learning finds hidden patterns or intrinsic structures in input data without labeled examples.",
        "Reinforcement learning is an area of machine learning where an agent learns to make decisions by taking actions in an environment to maximize reward.",
        
        # NLP 和语言模型
        "Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language in a valuable way.",
        "Large Language Models (LLMs) are AI systems trained on vast amounts of text data to understand and generate human-like text.",
        "BERT (Bidirectional Encoder Representations from Transformers) uses bidirectional training to understand context from both directions.",
        "GPT (Generative Pre-trained Transformer) is an autoregressive language model that generates coherent text by predicting the next word.",
        "Transformer architecture uses self-attention mechanisms to process sequential data more efficiently than RNNs or CNNs.",
        "Word embeddings represent words as dense vectors in a continuous vector space where semantically similar words are close together.",
        "Tokenization is the process of breaking down text into smaller units called tokens, which can be words, subwords, or characters.",
        
        # 搜索和检索
        "Information Retrieval (IR) is the activity of obtaining information system resources that are relevant to an information need from a collection.",
        "Vector search uses high-dimensional vectors to represent and search for similar items based on semantic meaning rather than exact matches.",
        "Semantic search understands the intent and contextual meaning of search queries rather than just matching keywords.",
        "Dense retrieval uses neural networks to encode queries and documents into dense vector representations for semantic similarity matching.",
        "Sparse retrieval methods like TF-IDF and BM25 rely on exact term matching and statistical analysis of word frequencies.",
        "Hybrid search combines dense and sparse retrieval methods to leverage both semantic understanding and exact term matching.",
        "Vector databases are specialized databases designed to store, index, and search high-dimensional vector embeddings efficiently.",
        "Approximate Nearest Neighbor (ANN) algorithms enable fast similarity search in high-dimensional vector spaces.",
        
        # 数据和存储
        "Big Data refers to extremely large datasets that require specialized tools and techniques for storage, processing, and analysis.",
        "Data preprocessing involves cleaning, transforming, and preparing raw data for analysis or machine learning models.",
        "Feature engineering is the process of selecting, modifying, or creating new features from raw data to improve model performance.",
        "Data pipelines automate the flow of data from source systems through various processing stages to final destinations.",
        "ETL (Extract, Transform, Load) processes move and transform data from source systems to target systems like data warehouses.",
        "Data lakes store raw, unstructured data in its native format, allowing for flexible processing and analysis later.",
        "Data warehouses are centralized repositories that store structured data from multiple sources optimized for analytical queries.",
        
        # 云计算和基础设施
        "Cloud computing delivers computing services including servers, storage, databases, networking, and software over the internet.",
        "Microservices architecture decomposes applications into small, independent services that communicate through well-defined APIs.",
        "Containerization packages applications and their dependencies into portable containers that can run consistently across environments.",
        "Kubernetes is an open-source container orchestration platform that automates deployment, scaling, and management of containerized applications.",
        "Serverless computing allows developers to run code without provisioning or managing servers, automatically scaling based on demand.",
        "API (Application Programming Interface) defines the methods and data formats that applications can use to communicate with each other.",
        "Load balancing distributes incoming network traffic across multiple servers to ensure high availability and optimal performance.",
        "Auto-scaling automatically adjusts computing resources up or down based on demand to maintain performance while optimizing costs."
    ]
    
    # 为每个文档添加元数据
    metadata = []
    categories = {
        "AI/ML": list(range(0, 7)),
        "NLP": list(range(7, 14)),
        "Search/Retrieval": list(range(14, 22)),
        "Data": list(range(22, 29)),
        "Cloud/Infrastructure": list(range(29, 36))
    }
    
    for i, doc in enumerate(tech_docs):
        category = None
        for cat, indices in categories.items():
            if i in indices:
                category = cat
                break
        
        metadata.append({
            "id": i,
            "category": category,
            "word_count": len(doc.split()),
            "added_at": datetime.now().isoformat()
        })
    
    return tech_docs, metadata

def main():
    """主函数 - 演示高级检索系统"""
    print("🚀 初始化高级文档检索系统...")
    retrieval_system = AdvancedRetrievalSystem()
    
    # 创建知识库
    docs, metadata = create_comprehensive_knowledge_base()
    retrieval_system.add_documents(docs, metadata)
    
    # 显示系统统计
    stats = retrieval_system.get_statistics()
    print(f"\n📊 系统统计:")
    print(f"   文档数量: {stats['documents']}")
    print(f"   嵌入维度: {stats['embedding_dim']}")
    print(f"   平均文档长度: {stats['avg_doc_length']:.1f} 词")
    print(f"   使用模型: {stats['model']}")
    
    # 测试查询
    test_queries = [
        "What is the difference between supervised and unsupervised learning?",
        "How do transformer models work in natural language processing?",
        "What are the advantages of microservices architecture?",
        "Explain vector databases and their use cases",
        "What is the difference between data lakes and data warehouses?"
    ]
    
    print(f"\n🔍 执行 {len(test_queries)} 个测试查询...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"🔎 查询 {i}: {query}")
        print('='*80)
        
        # 搜索相关文档
        results = retrieval_system.search(query, top_k=2, threshold=0.3)
        
        if not results:
            print("❌ 未找到相关文档")
            continue
        
        print(f"✅ 找到 {len(results)} 个相关文档:")
        
        for rank, (idx, similarity, doc, meta) in enumerate(results, 1):
            print(f"\n📄 排名 {rank} (相似度: {similarity:.4f})")
            print(f"   类别: {meta.get('category', 'Unknown')}")
            print(f"   文档: {doc}")
            if len(doc) > 100:
                print(f"   长度: {meta.get('word_count', 'Unknown')} 词")
    
    print(f"\n{'='*80}")
    print("🎉 检索演示完成！")
    print('='*80)

if __name__ == "__main__":
    main()
