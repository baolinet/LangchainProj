from langchain_ollama import OllamaLLM  # 更新导入路径
llm = OllamaLLM(model="llama3:8b")
# print(llm.invoke("你好"))

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{text}"),
])

# chain = prompt | llm
# result = chain.invoke({"input_language": "English", "output_language": "简体中文", "text": "I love programming."})
# print(result)

from langchain_core.output_parsers import StrOutputParser
output = StrOutputParser()
chain = prompt | llm | output
result = chain.invoke({"input_language": "English", "output_language": "简体中文", "text": "I love programming."})
print(result)