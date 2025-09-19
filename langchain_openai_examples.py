# LangChain OpenAI 完整示例
# pip install langchain-openai langchain-core

from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

# 配置信息
API_KEY = "115925abb19ec543cdcbe8af4506ff463ea2b5e8"
BASE_URL = "https://api-k2k2a5teg1h0idd9.aistudio-app.com/v1"
MODEL_NAME = "qwq"

def example_1_basic_chat():
    """示例1: 基础聊天功能"""
    print("🔥 示例1: 基础聊天功能")
    print("="*50)
    
    # 初始化ChatOpenAI
    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model=MODEL_NAME,
        temperature=0.7
    )
    
    # 单轮对话
    message = HumanMessage(content="请用一句话介绍什么是人工智能")
    response = llm.invoke([message])
    print(f"🤖 AI回复: {response.content}")
    print()

def example_2_system_prompt():
    """示例2: 使用系统提示"""
    print("🔥 示例2: 使用系统提示")
    print("="*50)
    
    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model=MODEL_NAME,
        temperature=0.5
    )
    
    # 多轮对话，包含系统提示
    messages = [
        SystemMessage(content="你是一位专业的Python编程导师，请用简洁明了的方式回答问题。"),
        HumanMessage(content="如何在Python中创建一个列表？")
    ]
    
    response = llm.invoke(messages)
    print(f"🐍 Python导师: {response.content}")
    print()

def example_3_streaming():
    """示例3: 流式输出"""
    print("🔥 示例3: 流式输出")
    print("="*50)
    
    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model=MODEL_NAME,
        temperature=0.6,
        streaming=True
    )
    
    message = HumanMessage(content="请详细解释什么是机器学习，包括其主要类型")
    
    print("🤖 AI正在回复...")
    print("-" * 30)
    
    try:
        for chunk in llm.stream([message]):
            print(chunk.content, end="", flush=True)
        print("\n" + "-" * 30)
        print("✅ 流式输出完成")
    except Exception as e:
        print(f"❌ 流式输出失败: {e}")
    print()

def example_4_prompt_template():
    """示例4: 使用提示模板"""
    print("🔥 示例4: 使用提示模板")
    print("="*50)
    
    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model=MODEL_NAME,
        temperature=0.7
    )
    
    # 创建聊天提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位{profession}，请用专业的角度回答问题。"),
        ("human", "{question}")
    ])
    
    # 创建链
    chain = prompt | llm | StrOutputParser()
    
    # 调用链
    result = chain.invoke({
        "profession": "营养师",
        "question": "如何制定健康的饮食计划？"
    })
    
    print(f"🥗 营养师建议: {result}")
    print()

def example_5_conversation():
    """示例5: 多轮对话"""
    print("🔥 示例5: 多轮对话")
    print("="*50)
    
    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model=MODEL_NAME,
        temperature=0.6
    )
    
    # 模拟多轮对话
    conversation = [
        SystemMessage(content="你是一位友好的AI助手。"),
        HumanMessage(content="你好！"),
    ]
    
    # 第一轮
    response1 = llm.invoke(conversation)
    conversation.append(response1)
    print(f"👤 用户: 你好！")
    print(f"🤖 AI: {response1.content}")
    
    # 第二轮
    conversation.append(HumanMessage(content="你能帮我解释一下什么是深度学习吗？"))
    response2 = llm.invoke(conversation)
    print(f"👤 用户: 你能帮我解释一下什么是深度学习吗？")
    print(f"🤖 AI: {response2.content}")
    print()

def example_6_legacy_openai():
    """示例6: 使用传统OpenAI接口（文本完成）"""
    print("🔥 示例6: 传统OpenAI接口（文本完成）")
    print("="*50)
    
    # 注意：这里使用OpenAI而不是ChatOpenAI
    llm = OpenAI(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model=MODEL_NAME,
        temperature=0.7,
        max_tokens=100
    )
    
    prompt = "请简要介绍Python编程语言的特点："
    
    try:
        response = llm.invoke(prompt)
        print(f"📝 文本完成: {response}")
    except Exception as e:
        print(f"❌ 文本完成失败: {e}")
        print("💡 提示: 某些API可能不支持传统的文本完成接口")
    print()

def example_7_batch_processing():
    """示例7: 批量处理"""
    print("🔥 示例7: 批量处理")
    print("="*50)
    
    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model=MODEL_NAME,
        temperature=0.5
    )
    
    # 批量问题
    questions = [
        "什么是人工智能？",
        "什么是机器学习？",
        "什么是深度学习？"
    ]
    
    print("🔄 批量处理问题...")
    for i, question in enumerate(questions, 1):
        message = HumanMessage(content=f"请用一句话回答：{question}")
        response = llm.invoke([message])
        print(f"{i}. 问题: {question}")
        print(f"   回答: {response.content}")
        time.sleep(1)  # 避免请求过快
    print()

def main():
    """主函数 - 运行所有示例"""
    print("🚀 LangChain OpenAI 完整示例集")
    print("🌟 展示各种使用方式和最佳实践")
    print("="*60)
    print()
    
    examples = [
        example_1_basic_chat,
        example_2_system_prompt,
        example_3_streaming,
        example_4_prompt_template,
        example_5_conversation,
        example_6_legacy_openai,
        example_7_batch_processing
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"❌ 示例{i}执行失败: {e}")
            print()
        
        if i < len(examples):
            print("⏳ 等待2秒后继续下一个示例...")
            time.sleep(2)
            print()
    
    print("🎉 所有示例执行完成！")
    print("="*60)

if __name__ == "__main__":
    main()
