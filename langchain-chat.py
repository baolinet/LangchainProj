from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,trim_messages

# llm = ChatOpenAI(
#     openai_api_key="115925abb19ec543cdcbe8af4506ff463ea2b5e8",
#     openai_api_base="https://api-77aaidn1l8c5b7xa.aistudio-app.com/v1",
#     model="gemma3:27b"
# )

# llm = OllamaLLM(model="llama3:8b")
from langchain_ollama import OllamaEmbeddings  # 新的导入方式
embeddings = OllamaEmbeddings(model="bge-m3:567m")


def simple_token_counter(messages):
    # 简单统计所有消息的词数
    return sum(len(msg.content.split()) for msg in messages)


trimmer = trim_messages(
    max_tokens=25,
    strategy="last",
    token_counter=simple_token_counter,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

print(trimmer.invoke(messages))

from langchain_openai import OpenAIEmbeddings

model = OpenAIEmbeddings(
    openai_api_key="115925abb19ec543cdcbe8af4506ff463ea2b5e8",
    openai_api_base="https://api-77aaidn1l8c5b7xa.aistudio-app.com/v1",
    model="text-embedding-3-small")

# llm = ChatOpenAI(
#     openai_api_key="115925abb19ec543cdcbe8af4506ff463ea2b5e8",
#     openai_api_base="https://api-77aaidn1l8c5b7xa.aistudio-app.com/v1",
#     model="gemma3:27b"
# )

model.embed_documents(["Hello, world!", "Goodbye, world!"])