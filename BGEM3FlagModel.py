# 使用ollama中的bge-m3模型 - 兼容原BGEM3FlagModel接口
from BGEM3FlagModel_compatible import OllamaBGEM3FlagModel
import numpy as np

# 直接替换原来的BGEM3FlagModel，接口完全兼容
model = OllamaBGEM3FlagModel('bge-m3:567m', use_fp16=True)

# 扩展文档库 - 包含更多技术文档
docs = [
    # 搜索和检索相关
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
    "Dense retrieval uses neural networks to encode queries and documents into dense vector representations for semantic matching.",
    "Sparse retrieval methods like TF-IDF and BM25 rely on exact term matching and statistical word frequencies.",
    "Vector databases store high-dimensional embeddings and provide efficient similarity search capabilities.",
    "Semantic search understands the meaning and context of queries rather than just matching keywords.",

    # 机器学习和AI相关
    "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
    "Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
    "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
    "Transformer architecture revolutionized NLP by introducing self-attention mechanisms for processing sequential data.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model that understands context from both directions.",
    "Large Language Models (LLMs) are AI systems trained on vast amounts of text data to generate human-like responses.",

    # 数据库和存储相关
    "Relational databases organize data into tables with rows and columns, using SQL for querying and management.",
    "NoSQL databases provide flexible data models for handling unstructured data, including document, key-value, and graph databases.",
    "Data warehouses are centralized repositories that store large amounts of structured data from multiple sources for analytics.",
    "ETL (Extract, Transform, Load) processes move data from source systems to data warehouses or other target systems.",
    "Data lakes store raw data in its native format, allowing for flexible processing and analysis later.",
    "OLAP (Online Analytical Processing) systems are designed for complex analytical queries and business intelligence.",

    # 编程和开发相关
    "Python is a high-level programming language known for its simplicity and extensive libraries for data science and AI.",
    "APIs (Application Programming Interfaces) define how different software components communicate with each other.",
    "REST (Representational State Transfer) is an architectural style for designing web services using HTTP methods.",
    "Microservices architecture breaks down applications into small, independent services that communicate over networks.",
    "Docker containers package applications and their dependencies into portable, lightweight units for deployment.",
    "Kubernetes orchestrates containerized applications, managing deployment, scaling, and operations automatically.",

    # 云计算和基础设施
    "Cloud computing delivers computing services over the internet, including servers, storage, databases, and software.",
    "AWS (Amazon Web Services) is a comprehensive cloud platform offering over 200 services for computing, storage, and databases.",
    "Serverless computing allows developers to run code without managing servers, automatically scaling based on demand.",
    "CDN (Content Delivery Network) distributes content across multiple geographic locations to improve performance.",
    "Load balancing distributes incoming network traffic across multiple servers to ensure high availability and performance.",
    "Auto-scaling automatically adjusts computing resources based on demand to maintain performance and optimize costs."
]

# 测试查询 - 涵盖不同主题
queries = [
    "What is the difference between dense and sparse retrieval?",
    "How do transformer models work in NLP?",
    "What are the benefits of using microservices architecture?",
    "Explain the concept of serverless computing",
    "What is the purpose of vector databases?"
]

print("=== 文档检索系统演示 ===")
print(f"文档库大小: {len(docs)} 个文档")
print(f"查询数量: {len(queries)} 个查询")
print()

# 生成文档嵌入向量
print("正在生成文档嵌入向量...")
docs_embeddings = model.encode(docs, batch_size=12, max_length=8192)['dense_vecs']
print(f"文档嵌入向量形状: {docs_embeddings.shape}")

# 对每个查询进行检索
for i, query in enumerate(queries):
    print(f"\n{'='*60}")
    print(f"查询 {i+1}: {query}")
    print('='*60)

    # 生成查询嵌入向量
    query_embedding = model.encode([query])['dense_vecs']

    # 计算相似度
    similarities = query_embedding @ docs_embeddings.T
    similarities = similarities[0]  # 取第一个查询的结果

    # 获取最相似的2个文档的索引
    top_indices = np.argsort(similarities)[::-1][:2]

    print("🔍 检索到的最相关文档 (Top 2):")
    for rank, idx in enumerate(top_indices, 1):
        similarity_score = similarities[idx]
        print(f"\n📄 排名 {rank} (相似度: {similarity_score:.4f}):")
        print(f"   {docs[idx]}")

print(f"\n{'='*60}")
print("检索任务完成！")
print('='*60)