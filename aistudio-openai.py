'''
# pip install openai
from openai import OpenAI

client = OpenAI(
    api_key="115925abb19ec543cdcbe8af4506ff463ea2b5e8",
    base_url="https://api-77aaidn1l8c5b7xa.aistudio-app.com/v1"
)

completion = client.chat.completions.create(
    model="gemma3:27b",
    temperature=0.6,
    messages=[
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    stream=True
)

for chunk in completion:
    if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
    else:
        print(chunk.choices[0].delta.content, end="", flush=True)
'''

# 通过fastap封装接口

# pip install fastapi uvicorn pydantic
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import json
from openai import OpenAI

# 创建FastAPI应用
app = FastAPI(title="AI Studio OpenAI API", description="基于百度AI Studio的OpenAI兼容API封装")

# 定义请求模型
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None

# 定义响应模型
class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

# 创建OpenAI客户端函数
def get_openai_client():
    return OpenAI(
        api_key="115925abb19ec543cdcbe8af4506ff463ea2b5e8",
        base_url="https://api-77aaidn1l8c5b7xa.aistudio-app.com/v1"
    )

# 健康检查端点
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "服务正常运行"}

# 聊天完成端点
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    client = get_openai_client()

    try:
        # 处理流式响应
        if request.stream:
            async def generate_stream():
                completion = client.chat.completions.create(
                    model=request.model,
                    messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
                    temperature=request.temperature,
                    stream=True
                )

                for chunk in completion:
                    if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                        # 构建流式响应格式
                        data = {
                            "id": chunk.id,
                            "object": "chat.completion.chunk",
                            "created": chunk.created,
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk.choices[0].delta.content},
                                "finish_reason": chunk.choices[0].finish_reason
                            }]
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                # 发送结束标记
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # 处理非流式响应
        else:
            completion = client.chat.completions.create(
                model=request.model,
                messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
                temperature=request.temperature,
                stream=False
            )

            return {
                "id": completion.id,
                "object": "chat.completion",
                "created": completion.created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion.choices[0].message.content
                    },
                    "finish_reason": completion.choices[0].finish_reason
                }],
                "usage": {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens
                }
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调用AI Studio API时出错: {str(e)}")

# 模型列表端点
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "gemma3:27b",
                "object": "model",
                "created": 1698969600,
                "owned_by": "baidu"
            },
            {
                "id": "gemma3:8b",
                "object": "model",
                "created": 1698969600,
                "owned_by": "baidu"
            }
        ]
    }

# 启动服务器的代码（取消注释即可运行）

if __name__ == "__main__":
    uvicorn.run("aistudio-openai:app", host="0.0.0.0", port=8000, reload=True)

