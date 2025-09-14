import requests
import json


def get_weather(city, api_key=None):
    """
    调用天气查询API获取指定城市的天气信息

    参数:
        city (str): 城市名称，如'Beijing'或'Shanghai'
        api_key (str, optional): OpenWeatherMap的API密钥，如果没有提供将使用免费测试API

    返回:
        dict: 包含天气信息的字典
    """
    # 如果没有提供API密钥，使用示例URL（注意：实际使用时需要注册获取自己的API密钥）
    if api_key is None:
        print("警告：未提供API密钥，使用示例数据。实际使用时请在OpenWeatherMap注册获取API密钥。")
        # 使用示例URL模拟API调用
        url = f"https://api.example.com/weather?q={city}"

        # 模拟返回数据
        mock_data = {
            "location": city,
            "weather": "晴天",
            "temperature": "26°C",
            "humidity": "45%",
            "wind": "东北风 3级"
        }
        return mock_data

    # 实际API调用（使用OpenWeatherMap API）
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=zh_cn"

    try:
        response = requests.get(url)
        response.raise_for_status()  # 如果请求失败则抛出异常

        # 解析JSON响应
        weather_data = response.json()

        # 提取需要的天气信息
        result = {
            "城市": weather_data["name"],
            "天气": weather_data["weather"][0]["description"],
            "温度": f"{weather_data['main']['temp']}°C",
            "体感温度": f"{weather_data['main']['feels_like']}°C",
            "湿度": f"{weather_data['main']['humidity']}%",
            "风速": f"{weather_data['wind']['speed']} m/s"
        }
        return result

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP错误: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"连接错误: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"请求超时: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"请求异常: {req_err}")
    except KeyError as key_err:
        print(f"解析响应数据错误: {key_err}")
    except json.JSONDecodeError as json_err:
        print(f"JSON解析错误: {json_err}")

    return None


def display_weather(weather_data):
    """
    显示天气信息

    参数:
        weather_data (dict): 包含天气信息的字典
    """
    if weather_data is None:
        print("无法获取天气数据")
        return

    print("\n===== 天气信息 =====")
    for key, value in weather_data.items():
        print(f"{key}: {value}")
    print("==================\n")


# 使用示例
if __name__ == "__main__":
    # 1. 使用模拟数据（不需要API密钥）
    city = "北京"
    weather_data = get_weather(city)
    display_weather(weather_data)

    # 2. 使用实际API（需要替换为您的API密钥）
    # 取消下面注释并替换YOUR_API_KEY为您的OpenWeatherMap API密钥
    # api_key = "YOUR_API_KEY"
    # city = "上海"
    # weather_data = get_weather(city, api_key)
    # display_weather(weather_data)
