# 交互式文档检索系统
from advanced_retrieval_system import AdvancedRetrievalSystem, create_comprehensive_knowledge_base
import sys

def display_welcome():
    """显示欢迎信息"""
    print("🎯" + "="*70 + "🎯")
    print("🚀           交互式文档检索系统 (基于 Ollama BGE-M3)           🚀")
    print("🎯" + "="*70 + "🎯")
    print()
    print("📚 系统功能:")
    print("   • 语义搜索 - 理解查询的含义而不仅仅是关键词匹配")
    print("   • 多类别文档 - AI/ML, NLP, 搜索检索, 数据处理, 云计算")
    print("   • 相似度评分 - 显示每个结果的相关性分数")
    print("   • 分类标签 - 每个文档都有明确的类别标签")
    print()
    print("💡 使用提示:")
    print("   • 输入您的问题或查询")
    print("   • 输入 'quit' 或 'exit' 退出")
    print("   • 输入 'help' 查看帮助信息")
    print("   • 输入 'stats' 查看系统统计")
    print("   • 输入 'examples' 查看示例查询")
    print()

def display_help():
    """显示帮助信息"""
    print("\n📖 帮助信息:")
    print("="*50)
    print("🔍 查询类型:")
    print("   • 概念解释: 'What is machine learning?'")
    print("   • 比较分析: 'Difference between SQL and NoSQL'")
    print("   • 技术原理: 'How do neural networks work?'")
    print("   • 应用场景: 'When to use microservices?'")
    print()
    print("⚙️  系统命令:")
    print("   • help     - 显示此帮助信息")
    print("   • stats    - 显示系统统计信息")
    print("   • examples - 显示示例查询")
    print("   • quit/exit - 退出系统")
    print("="*50)

def display_examples():
    """显示示例查询"""
    examples = [
        "What is deep learning and how does it work?",
        "Explain the difference between supervised and unsupervised learning",
        "How do transformer models revolutionize NLP?",
        "What are the benefits of using vector databases?",
        "Compare microservices vs monolithic architecture",
        "What is serverless computing and its advantages?",
        "Explain the concept of data lakes vs data warehouses",
        "How does semantic search differ from keyword search?",
        "What is the role of APIs in modern applications?",
        "Describe the ETL process in data engineering"
    ]
    
    print("\n💡 示例查询:")
    print("="*50)
    for i, example in enumerate(examples, 1):
        print(f"{i:2d}. {example}")
    print("="*50)

def display_stats(retrieval_system):
    """显示系统统计信息"""
    stats = retrieval_system.get_statistics()
    print("\n📊 系统统计信息:")
    print("="*50)
    print(f"📄 文档总数:     {stats['documents']}")
    print(f"🧮 嵌入维度:     {stats['embedding_dim']}")
    print(f"📝 平均文档长度: {stats['avg_doc_length']:.1f} 词")
    print(f"🤖 使用模型:     {stats['model']}")
    print("="*50)

def format_results(results, query):
    """格式化搜索结果"""
    if not results:
        print("❌ 抱歉，没有找到相关的文档。")
        print("💡 建议:")
        print("   • 尝试使用不同的关键词")
        print("   • 使用更具体或更一般的查询")
        print("   • 输入 'examples' 查看示例查询")
        return
    
    print(f"✅ 为查询 '{query}' 找到 {len(results)} 个相关文档:")
    print("="*80)
    
    for rank, (idx, similarity, doc, meta) in enumerate(results, 1):
        # 相似度条形图
        bar_length = int(similarity * 20)  # 最大20个字符
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        print(f"\n🏆 排名 {rank}")
        print(f"📊 相似度: {similarity:.4f} [{bar}] {similarity*100:.1f}%")
        print(f"🏷️  类别: {meta.get('category', 'Unknown')}")
        print(f"📄 内容: {doc}")
        print(f"📏 长度: {meta.get('word_count', 'Unknown')} 词")
        
        if rank < len(results):
            print("-" * 80)

def main():
    """主交互循环"""
    display_welcome()
    
    # 初始化检索系统
    print("🔄 正在初始化检索系统...")
    retrieval_system = AdvancedRetrievalSystem()
    
    # 加载知识库
    docs, metadata = create_comprehensive_knowledge_base()
    retrieval_system.add_documents(docs, metadata)
    
    print("✅ 系统初始化完成！")
    print("\n" + "="*70)
    print("💬 请输入您的查询 (输入 'help' 获取帮助):")
    
    query_count = 0
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n🔍 查询> ").strip()
            
            # 处理空输入
            if not user_input:
                print("⚠️  请输入有效的查询或命令")
                continue
            
            # 处理系统命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 感谢使用文档检索系统！再见！")
                break
            elif user_input.lower() == 'help':
                display_help()
                continue
            elif user_input.lower() == 'stats':
                display_stats(retrieval_system)
                continue
            elif user_input.lower() == 'examples':
                display_examples()
                continue
            
            # 执行搜索
            query_count += 1
            print(f"\n🔎 正在搜索第 {query_count} 个查询...")
            
            results = retrieval_system.search(user_input, top_k=3, threshold=0.2)
            format_results(results, user_input)
            
            print("\n" + "="*70)
            
        except KeyboardInterrupt:
            print("\n\n👋 检测到 Ctrl+C，正在退出...")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("💡 请重试或输入 'help' 获取帮助")

if __name__ == "__main__":
    main()
