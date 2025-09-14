import os
os.environ["DASHSCOPE_API_KEY"] = "sk-8cfd64ba9a994c74bea6e4a7184bbd70"
from langchain_community.llms import Tongyi
from langchain_core.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage(content="Translate the following from English into Chinese"),
    HumanMessage(content="hi!"),
]
Tongyi().invoke(messages)