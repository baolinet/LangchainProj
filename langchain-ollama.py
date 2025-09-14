# 原代码（已注释）
# import getpass
# import os
#
# if not os.environ.get("OPENAI_API_KEY"):
#   os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
#
# from langchain.chat_models import init_chat_model
#
# model = init_chat_model("gpt-4o-mini", model_provider="openai")

# 新代码：使用Ollama的LangChain示例

"""
Ollama + LangChain 示例代码

运行前需要安装以下依赖包：
pip install langchain langchain-community faiss-cpu

还需要安装并运行Ollama：
1. 从 https://ollama.com/ 下载并安装Ollama
2. 运行Ollama并拉取模型： ollama pull llama3
"""

# 导入必要的库
try:
    # 尝试导入必要的库
    from langchain.llms import Ollama
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import OllamaEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains.question_answering import load_qa_chain
except ImportError:
    print("错误: 无法导入LangChain库。请确保已安装必要的依赖包。")
    print("请运行: pip install langchain langchain-community faiss-cpu")
    import sys
    sys.exit(1)

# 定义模型名称
# 可以替换为其他已安装的模型，如 "mistral", "gemma", "llama2" 等
MODEL_NAME = "llama3:8b"

def simple_llm_example():
    """简单的LLM调用示例"""
    print("\n=== 简单LLM调用示例 ===")

    try:
        # 初始化Ollama LLM
        # 注意：需要确保Ollama服务正在运行，并且已安装了指定的模型
        llm = Ollama(model=MODEL_NAME)

        # 直接调用LLM
        print(f"发送提示到Ollama ({MODEL_NAME})...")
        response = llm.invoke("解释一下什么是人工智能，用三句话概括")
        print(f"LLM响应:\n{response}\n")
        return True
    except Exception as e:
        print(f"运行简单LLM示例时出错: {e}")
        return False

def chain_example():
    """使用LangChain链的示例"""
    print("\n=== LangChain链示例 ===")

    try:
        # 初始化Ollama LLM
        llm = Ollama(model=MODEL_NAME)

        # 创建一个提示模板
        template = """你是一位专业的{profession}。
        请针对以下问题提供专业的建议：{question}
        """

        prompt = PromptTemplate(
            input_variables=["profession", "question"],
            template=template,
        )

        # 创建LLMChain
        chain = LLMChain(llm=llm, prompt=prompt)

        # 运行链
        print(f"发送提示到LangChain链...")
        result = chain.invoke({
            "profession": "营养师",
            "question": "如何保持健康的饮食习惯？"
        })

        print(f"链响应:\n{result['text']}\n")
        return True
    except Exception as e:
        print(f"运行LangChain链示例时出错: {e}")
        return False

def rag_example():
    """检索增强生成(RAG)示例"""
    print("\n=== 检索增强生成(RAG)示例 ===")

    try:
        # 示例文档
        documents = [
            "人工智能(AI)是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
            "机器学习是AI的一个子领域，专注于开发能从数据中学习的算法。",
            "深度学习是机器学习的一种特殊形式，使用神经网络进行学习。",
            "自然语言处理(NLP)使计算机能够理解、解释和生成人类语言。",
            "计算机视觉使机器能够从图像或视频中获取信息并进行解释。"
        ]

        print("准备文档和向量存储...")
        # 文本分割
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        # texts = text_splitter.split_documents([{"page_content": doc} for doc in documents])
        texts = text_splitter.split_documents([{"page_content": doc} for doc in documents])  # 修复：使用 split_documents 方法

        # 创建向量存储
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
        vectorstore = FAISS.from_documents(texts, embeddings)

        # 创建问答链
        llm = Ollama(model=MODEL_NAME)
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        # 用户查询
        query = "什么是深度学习？"

        # 检索相关文档
        print(f"执行查询: '{query}'")
        docs = vectorstore.similarity_search(query)

        # 生成回答
        print("生成RAG回答...")
        response = qa_chain.invoke({"input_documents": docs, "question": query})

        print(f"查询: {query}")
        print(f"检索到的文档: {[doc.page_content for doc in docs]}")
        print(f"RAG响应: {response['output_text']}\n")
        return True
    except Exception as e:
        print(f"运行RAG示例时出错: {e}")
        return False

# 主函数
def main():
    print("\n========================================")
    print("     Ollama + LangChain 示例     ")
    print("========================================\n")
    print(f"使用模型: {MODEL_NAME}")
    print("注意: 请确保Ollama服务正在运行，并且已安装所需模型")
    print("\n如果需要安装Ollama，请访问: https://ollama.com/")
    print(f"如果需要安装模型，请运行: ollama pull {MODEL_NAME}\n")

    # 运行示例
    success_count = 0
    total_examples = 3

    # 运行简单LLM示例
    if simple_llm_example():
        success_count += 1

    # 运行LangChain链示例
    if chain_example():
        success_count += 1

    # 运行RAG示例
    if rag_example():
        success_count += 1

    # 显示汇总信息
    print("\n========================================")
    print(f"示例运行完成: {success_count}/{total_examples} 成功")

    if success_count < total_examples:
        print("\n如果遇到问题，请确保:")
        print(f"1. Ollama服务正在运行 (通常在 http://localhost:11434)")
        print(f"2. 已安装 {MODEL_NAME} 模型 (可以运行 'ollama pull {MODEL_NAME}' 安装)")
        print(f"3. 已安装必要的Python库 (pip install langchain langchain-community faiss-cpu)")
    else:
        print("\n所有示例都成功运行了！")

    print("========================================\n")

if __name__ == "__main__":
    main()