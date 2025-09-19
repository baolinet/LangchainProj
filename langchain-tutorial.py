# pip install langchain-openai
import sys
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.messages import HumanMessage

print("🔍 测试 OpenAI vs ChatOpenAI 的区别")
print("="*50)

# 方法1: 使用 ChatOpenAI (推荐，适用于现代聊天API)
print("\n✅ 方法1: 使用 ChatOpenAI (推荐)")
try:
    chat_llm = ChatOpenAI(
        openai_api_key="115925abb19ec543cdcbe8af4506ff463ea2b5e8",
        openai_api_base="https://api-77aaidn1l8c5b7xa.aistudio-app.com/v1",
        model="gemma3:27b",
        temperature=0.6
    )

    # ChatOpenAI 需要使用消息格式
    messages = [HumanMessage(content="你好，请介绍一下自己")]
    response = chat_llm.invoke(messages)
    print(f"🤖 ChatOpenAI 回复: {response.content}")

except Exception as e:
    print(f"❌ ChatOpenAI 失败: {e}")

# 方法2: 使用 OpenAI (传统文本完成API，可能不被支持)
print("\n⚠️  方法2: 使用 OpenAI (传统API，可能失败)")
try:
    text_llm = OpenAI(
        openai_api_key="115925abb19ec543cdcbe8af4506ff463ea2b5e8",
        openai_api_base="https://api-77aaidn1l8c5b7xa.aistudio-app.com/v1",
        model="gemma3:27b",
        temperature=0.6
    )

    # OpenAI 直接接受字符串
    response = text_llm.invoke("你好，请介绍一下自己")
    print(f"📝 OpenAI 回复: {response}")

except Exception as e:
    print(f"❌ OpenAI 失败: {e}")
    print("💡 这是正常的，因为您的API可能不支持传统的文本完成端点")

print("\n" + "="*50)
print("📚 总结:")
print("• ChatOpenAI: 使用 /chat/completions 端点，支持对话格式")
print("• OpenAI: 使用 /completions 端点，支持文本完成")
print("• 现代API通常只支持 ChatOpenAI 格式")
print("="*50)

