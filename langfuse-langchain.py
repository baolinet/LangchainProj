from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import HumanMessage
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate

langfuse = Langfuse(
    public_key="pk-lf-7f5506e1-adf4-429e-9a27-e08df85d27c0",
    secret_key="sk-lf-32ec21ab-f7d4-4f66-bb29-004df70a7a3a",
    host="http://localhost:3000"
)

langfuse_handler = CallbackHandler()

# <Your Langchain code here>
llm = ChatOpenAI(
    openai_api_key="115925abb19ec543cdcbe8af4506ff463ea2b5e8",
    openai_api_base="https://api-77aaidn1l8c5b7xa.aistudio-app.com/v1",
    model="gemma3:27b"
)

prompt_template = """
我的名字叫【{name}】，我的个人介绍是【{description}】。
请根据我的名字和介绍，帮我想一段有吸引力的自我介绍的句子，以此来吸引读者关注和点赞我的账号。
"""
prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(prompt_template)
])

# 定义输出解析器
parser = StrOutputParser()

chain = (
    {"name":RunnablePassthrough(), "description":RunnablePassthrough() } ## 这里是给prompt的输入，两个参数
    | prompt
    | llm
    | parser
)

## invoke的第一个参数，传入json格式的参数，key与prompt中的参数名一致
# response = chain.invoke({'name': 'QA', 'description': '互联网公司功能测试人员'}, config={"callbacks":[langfuse_handler]})
# print(response)



import asyncio

async def test():
    chain2 = prompt | llm  # 使用管道操作符组合
    async for chunk in chain2.astream({'name': '小刘', 'description': '市场销售'}, config={"callbacks": [langfuse_handler]}):
        print(getattr(chunk, "content", chunk), end="")  # 不自动换行
    print()  # 最后补一个换行
    
if __name__ == "__main__":
    asyncio.run(test())