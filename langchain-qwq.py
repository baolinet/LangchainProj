import os
os.environ["DASHSCOPE_API_KEY"] = "sk-b13ee183c99b48ecb93d0b007e4205e1"
from langchain_qwq import ChatQwQ

llm = ChatQwQ(
    model="qwq-plus",
    timeout=None,
    max_retries=2
)

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French."
#         "Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)

# chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that translates"
            "{input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)

#tools
from langchain_core.tools import tool
from langchain_qwq import ChatQwQ


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


llm = ChatQwQ()

llm_with_tools = llm.bind_tools([multiply])

msg = llm_with_tools.invoke("What's 5 times forty two")

print(msg)