# ChatOpenAI 最佳实践示例
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def create_llm():
    """创建ChatOpenAI实例 - 推荐方式"""
    return ChatOpenAI(
        openai_api_key="115925abb19ec543cdcbe8af4506ff463ea2b5e8",
        openai_api_base="https://api-77aaidn1l8c5b7xa.aistudio-app.com/v1",
        model="gemma3:27b",
        temperature=0.6
    )

def simple_chat():
    """简单聊天示例 - 正确的方式"""
    print("💬 简单聊天示例")
    print("-" * 30)
    
    llm = create_llm()
    
    # ✅ 正确：使用消息格式
    message = HumanMessage(content="你好，请用一句话介绍自己")
    response = llm.invoke([message])
    print(f"🤖 AI: {response.content}")
    
    # ❌ 错误示例（会报错）：
    # response = llm.invoke("你好，请用一句话介绍自己")  # 这样会报错

def system_prompt_chat():
    """带系统提示的聊天 - ChatOpenAI的优势"""
    print("\n🎭 系统提示示例")
    print("-" * 30)
    
    llm = create_llm()
    
    # ✅ 使用系统提示设定角色
    messages = [
        SystemMessage(content="你是一位专业的Python编程助手，回答要简洁明了"),
        HumanMessage(content="什么是装饰器？")
    ]
    response = llm.invoke(messages)
    print(f"🐍 Python助手: {response.content}")

def streaming_example():
    """流式输出示例"""
    print("\n🌊 流式输出示例")
    print("-" * 30)
    
    # 启用流式输出
    llm = ChatOpenAI(
        openai_api_key="115925abb19ec543cdcbe8af4506ff463ea2b5e8",
        openai_api_base="https://api-77aaidn1l8c5b7xa.aistudio-app.com/v1",
        model="gemma3:27b",
        temperature=0.6,
        streaming=True
    )
    
    message = HumanMessage(content="请简要解释什么是机器学习")
    
    print("🤖 AI正在回复...")
    try:
        for chunk in llm.stream([message]):
            print(chunk.content, end="", flush=True)
        print("\n✅ 流式输出完成")
    except Exception as e:
        print(f"\n❌ 流式输出失败: {e}")

def error_handling_example():
    """错误处理示例"""
    print("\n🛡️ 错误处理示例")
    print("-" * 30)
    
    # 演示如何优雅地处理错误
    try:
        llm = create_llm()
        message = HumanMessage(content="测试连接")
        response = llm.invoke([message])
        print(f"✅ 连接成功: {response.content[:50]}...")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("💡 建议检查:")
        print("   - API密钥是否正确")
        print("   - 网络连接是否正常")
        print("   - API端点是否支持ChatOpenAI格式")

def main():
    """主函数"""
    print("🚀 ChatOpenAI 最佳实践示例")
    print("="*50)
    print("📚 学习要点:")
    print("   • 使用 ChatOpenAI 而不是 OpenAI")
    print("   • 输入必须是消息列表格式")
    print("   • 输出需要访问 .content 属性")
    print("   • 支持系统提示和多轮对话")
    print("="*50)
    
    examples = [
        simple_chat,
        system_prompt_chat,
        streaming_example,
        error_handling_example
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ 示例执行失败: {e}")
    
    print(f"\n🎉 示例演示完成！")
    print("💡 记住：ChatOpenAI 是现代推荐方案，兼容性更好！")

if __name__ == "__main__":
    main()
