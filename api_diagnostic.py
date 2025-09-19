# API诊断工具
import requests
import json
from datetime import datetime

def test_api_endpoints():
    """测试不同的API端点"""
    
    base_url = "https://api-77aaidn1l8c5b7xa.aistudio-app.com/v1"
    api_key = "115925abb19ec543cdcbe8af4506ff463ea2b5e8"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print("🔍 API诊断工具")
    print("="*50)
    print(f"🌐 基础URL: {base_url}")
    print(f"🔑 API密钥: {api_key[:20]}...")
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    # 测试1: 模型列表
    print("\n📋 测试1: 获取模型列表")
    try:
        response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
        print(f"   状态码: {response.status_code}")
        print(f"   响应头: {dict(response.headers)}")
        if response.status_code == 200:
            models = response.json()
            print(f"   ✅ 成功获取模型列表")
            if 'data' in models:
                print(f"   📝 可用模型数量: {len(models['data'])}")
                for model in models['data'][:5]:  # 只显示前5个
                    print(f"      - {model.get('id', 'unknown')}")
            else:
                print(f"   📝 响应内容: {models}")
        else:
            print(f"   ❌ 失败: {response.text}")
    except Exception as e:
        print(f"   ❌ 异常: {e}")
    
    # 测试2: 聊天完成
    print("\n💬 测试2: 聊天完成接口")
    try:
        chat_data = {
            "model": "gemma3:27b",
            "messages": [
                {"role": "user", "content": "Hello, please respond in Chinese"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{base_url}/chat/completions", 
            headers=headers, 
            json=chat_data, 
            timeout=30
        )
        print(f"   状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ 聊天接口正常")
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0].get('message', {}).get('content', '')
                print(f"   📝 模型回复: {message}")
            else:
                print(f"   📝 完整响应: {result}")
        else:
            print(f"   ❌ 失败: {response.text}")
    except Exception as e:
        print(f"   ❌ 异常: {e}")
    
    # 测试3: 文本完成
    print("\n📝 测试3: 文本完成接口")
    try:
        completion_data = {
            "model": "gemma3:27b",
            "prompt": "请用中文回答：你好",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{base_url}/completions", 
            headers=headers, 
            json=completion_data, 
            timeout=30
        )
        print(f"   状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ 文本完成接口正常")
            if 'choices' in result and len(result['choices']) > 0:
                text = result['choices'][0].get('text', '')
                print(f"   📝 模型回复: {text}")
            else:
                print(f"   📝 完整响应: {result}")
        else:
            print(f"   ❌ 失败: {response.text}")
    except Exception as e:
        print(f"   ❌ 异常: {e}")
    
    # 测试4: 网络连通性
    print("\n🌐 测试4: 基础网络连通性")
    try:
        response = requests.get(base_url, timeout=10)
        print(f"   状态码: {response.status_code}")
        print(f"   ✅ 网络连接正常")
    except Exception as e:
        print(f"   ❌ 网络连接异常: {e}")
    
    print("\n" + "="*50)
    print("🎯 诊断建议:")
    print("   1. 如果所有测试都失败，检查网络连接")
    print("   2. 如果返回401/403，检查API密钥")
    print("   3. 如果返回404，检查API端点URL")
    print("   4. 如果返回422，检查请求参数格式")
    print("   5. 如果模型不存在，检查模型名称")
    print("="*50)

if __name__ == "__main__":
    test_api_endpoints()
