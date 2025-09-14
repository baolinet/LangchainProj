from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

api_key = "sk-lwulctgpjamiusubdokjnttswwckbyjmnkffoqmqfxechdpk"
base_url = "https://api.siliconflow.cn/v1"
model = init_chat_model("Qwen/Qwen2.5-72B-Instruct", model_provider="openai",api_key=api_key,base_url=base_url)

print(model.invoke([HumanMessage(content="你是谁？")]))

messages = [  
    HumanMessage("你是谁？"),
]
for token in model.stream(messages):
    print(token.content, end="|",flash=True)