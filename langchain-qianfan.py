import os
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.messages import HumanMessage

os.environ["QIANFAN_AK"] = "aW6X5OQn2K1Yz5CREvcjGwtQ"
os.environ["QIANFAN_SK"] = "6QUfkGKTDSuocPbmafs58XcNOWRkFY7t"

chat = QianfanChatEndpoint(
    streaming=True,
)
res = chat([HumanMessage(content="今天是星期几")])
print(res.content)