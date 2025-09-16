# 演示检索系统 - 展示具体的查询和结果
from BGEM3FlagModel_compatible import OllamaBGEM3FlagModel
import numpy as np

def create_demo_knowledge_base():
    """创建演示用的知识库"""
    return [
        # AI/ML 基础概念
        "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.",
        "Machine Learning is a method of data analysis that automates analytical model building using algorithms that iteratively learn from data.",
        "Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model complex patterns.",
        "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains.",
        "Supervised learning uses labeled examples to learn a mapping from inputs to outputs, like classification and regression tasks.",
        "Unsupervised learning finds hidden patterns in data without labeled examples, including clustering and dimensionality reduction.",
        
        # NLP 和语言技术
        "Natural Language Processing (NLP) combines computational linguistics with statistical and machine learning methods to help computers understand human language.",
        "Large Language Models (LLMs) are neural networks trained on massive text datasets to understand and generate human-like text.",
        "BERT (Bidirectional Encoder Representations from Transformers) reads text bidirectionally to better understand context and meaning.",
        "GPT (Generative Pre-trained Transformer) is an autoregressive language model that generates text by predicting the next word in a sequence.",
        "Transformers use self-attention mechanisms to process sequences of data, revolutionizing natural language processing tasks.",
        "Word embeddings represent words as dense vectors in a continuous vector space where similar words have similar representations.",
        
        # 搜索和检索技术
        "Information retrieval is the activity of obtaining information system resources that are relevant to an information need from a collection of information resources.",
        "Vector search uses mathematical vectors to represent data points and finds similar items by calculating distances in high-dimensional space.",
        "Semantic search understands the intent and contextual meaning behind search queries rather than just matching exact keywords.",
        "Dense retrieval encodes queries and documents into dense vector representations using neural networks for semantic similarity matching.",
        "Sparse retrieval methods like TF-IDF and BM25 use statistical analysis of term frequencies and exact keyword matching.",
        "Vector databases are specialized storage systems optimized for storing, indexing, and querying high-dimensional vector embeddings.",
        
        # 数据处理和存储
        "Big Data refers to datasets that are too large or complex for traditional data processing applications to handle effectively.",
        "Data preprocessing involves cleaning, transforming, and preparing raw data for analysis or machine learning model training.",
        "ETL (Extract, Transform, Load) is a data integration process that combines data from multiple sources into a single, consistent data store.",
        "Data warehouses are centralized repositories that store integrated data from multiple sources for business intelligence and analytics.",
        "Data lakes are storage repositories that hold vast amounts of raw data in its native format until it's needed for analysis.",
        "NoSQL databases provide flexible data models for storing and retrieving unstructured or semi-structured data at scale.",
        
        # 云计算和架构
        "Cloud computing delivers computing services including servers, storage, databases, networking, software, and analytics over the internet.",
        "Microservices architecture structures an application as a collection of loosely coupled services that communicate through APIs.",
        "Containerization packages software code with all its dependencies so the application runs quickly and reliably across computing environments.",
        "Kubernetes is an open-source system for automating deployment, scaling, and management of containerized applications.",
        "Serverless computing allows developers to build and run applications without thinking about servers or infrastructure management.",
        "APIs (Application Programming Interfaces) are sets of protocols and tools for building software applications and enabling communication between different systems."
    ]

def run_demo_queries():
    """运行演示查询"""
    print("🚀 初始化 BGE-M3 检索系统演示")
    print("="*60)
    
    # 初始化模型
    model = OllamaBGEM3FlagModel('bge-m3:567m', use_fp16=True)
    
    # 创建知识库
    docs = create_demo_knowledge_base()
    print(f"📚 知识库包含 {len(docs)} 个文档")
    
    # 生成文档嵌入
    print("🔄 正在生成文档嵌入向量...")
    doc_embeddings = model.encode(docs)['dense_vecs']
    print(f"✅ 完成！嵌入向量形状: {doc_embeddings.shape}")
    
    # 演示查询列表
    demo_queries = [
        {
            "query": "What is the difference between machine learning and deep learning?",
            "description": "比较机器学习和深度学习的区别"
        },
        {
            "query": "How do vector databases work for similarity search?",
            "description": "向量数据库如何进行相似性搜索"
        },
        {
            "query": "Explain the benefits of microservices architecture",
            "description": "解释微服务架构的优势"
        },
        {
            "query": "What are the main differences between data lakes and data warehouses?",
            "description": "数据湖和数据仓库的主要区别"
        },
        {
            "query": "How do transformer models work in natural language processing?",
            "description": "Transformer模型在自然语言处理中的工作原理"
        }
    ]
    
    print(f"\n🔍 开始执行 {len(demo_queries)} 个演示查询...")
    
    for i, query_info in enumerate(demo_queries, 1):
        query = query_info["query"]
        description = query_info["description"]
        
        print(f"\n{'='*80}")
        print(f"🔎 查询 {i}: {query}")
        print(f"📝 说明: {description}")
        print('='*80)
        
        # 生成查询嵌入
        query_embedding = model.encode([query])['dense_vecs']
        
        # 计算相似度
        similarities = query_embedding @ doc_embeddings.T
        similarities = similarities[0]
        
        # 获取前2个最相似的文档
        top_indices = np.argsort(similarities)[::-1][:2]
        
        print("🎯 检索结果 (Top 2):")
        
        for rank, idx in enumerate(top_indices, 1):
            similarity = similarities[idx]
            doc = docs[idx]
            
            # 创建相似度可视化
            bar_length = int(similarity * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            
            print(f"\n📄 排名 {rank}")
            print(f"📊 相似度: {similarity:.4f} [{bar}] ({similarity*100:.1f}%)")
            print(f"📖 文档内容:")
            print(f"   {doc}")
            
            # 高亮关键词（简单实现）
            query_words = query.lower().split()
            doc_words = doc.lower().split()
            common_words = set(query_words) & set(doc_words)
            if common_words:
                print(f"🔑 共同关键词: {', '.join(common_words)}")
    
    print(f"\n{'='*80}")
    print("🎉 演示完成！")
    print("💡 提示: 您可以运行 'python interactive_retrieval.py' 进行交互式查询")
    print('='*80)

if __name__ == "__main__":
    run_demo_queries()
