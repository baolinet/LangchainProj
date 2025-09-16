from langfuse import Langfuse

langfuse = Langfuse(
  secret_key="sk-lf-32ec21ab-f7d4-4f66-bb29-004df70a7a3a",
  public_key="pk-lf-7f5506e1-adf4-429e-9a27-e08df85d27c0",
  host="http://localhost:3000"
)

# langfuse.create_dataset(
#     name="FirstDataset",
#     description="My first dataset",
#     metadata={
#         "author": "Alice",
#         "date": "2022-01-01",
#         "type": "benchmark"
#     }
# )

langfuse.create_dataset_item(
    dataset_name="FirstDataset",
    input={
        "text": "hello world"
    },
    expected_output={
        "text": "hello world"
    },
    metadata={
        "model": "llama3",
    }
)

# # 调用 dify 接口发送消息，获取 dify 响应 
# async def send_chat_message(
#     query: str,
#     inputs: dict = {},
#     url: str = os.getenv("DIFY_API_BASE", ""),
#     api_key: str = os.getenv("DIFY_API_KEY", ""),
#     response_mode: Literal["streaming", "blocking"] = "blocking",
#     user: str = "auto_test",
#     file_array: list = [],
# ):
#     chat_url = f"{url}/chat-messages"
#     headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#     payload = {
#         "inputs": inputs,
#         "query": query,
#         "response_mode": response_mode,
#         "conversation_id": "",
#         "user": user,
#         "files": file_array,
#     }
#     async with aiohttp.ClientSession() as session:
#         async with session.post(chat_url, headers=headers, json=payload) as response:
#             ret = await response.json()
#             status = ret.get("status")
#             message = ret.get("message")
#             if status and message:
#                 raise ValueError(f"{status}: {message}")
#             return ret

# async def run_dataset_item(item, run_name):
#     response = await send_chat_message(item.input)

#     # 将 dify 返回的 message_id 与 langfuse 中的 trace id 进行关联 
#     item.link(
#         trace_or_observation=None,
#         run_name=run_name,
#         trace_id=response["message_id"],
#         observation_id=None,
#     )
#     return response

