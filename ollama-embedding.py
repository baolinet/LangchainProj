# import requests
# response = requests.post(
#     "http://localhost:11434/api/embeddings",
#     json={"model": "bge-m3:567m", "input": "你好，这是一个测试句子"}
# )
# embedding = response.json()["embedding"]  # 获取1024维向量

# print(embedding)


from langchain_ollama import OllamaEmbeddings  # 新的导入方式
embeddings = OllamaEmbeddings(model="bge-m3:567m")
text = "Hello World!"
embedding = embeddings.embed_query(text)  # 生成嵌入向量

print(embedding)