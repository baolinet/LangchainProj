import requests
import json
import sys


import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 将环境变量转换为字典并打印
env_vars = dict(os.environ)
# print(env_vars)

class Config:
    def __init__(self):
        # 初始化未加载的配置字段（用None标记未加载）
        self._api_key = None
        self._database_url = None
        self._timeout = None

    @property
    def api_key(self):
        # 首次访问时自动加载
        if self._api_key is None:
            self._load_api_key()
        return self._api_key

    @property
    def database_url(self):
        if self._database_url is None:
            self._load_database_url()
        return self._database_url

    @property
    def timeout(self):
        if self._timeout is None:
            self._load_timeout()
        return self._timeout

    # 实际加载配置的函数（可从.env、数据库、远程接口等来源加载）
    def _load_api_key(self):
        print("自动加载API密钥...")
        load_dotenv()  # 加载.env文件
        self._api_key = os.getenv("API_KEY", "default_api_key")

    def _load_database_url(self):
        print("自动加载数据库地址...")
        load_dotenv()
        self._database_url = os.getenv("DATABASE_URL", "sqlite:///default.db")

    def _load_timeout(self):
        print("自动加载超时配置...")
        # 示例：从配置文件加载（实际可替换为JSON/XML等格式）
        self._timeout = 30  # 默认值


# 使用示例
if __name__ == "__main__":
    config = Config()
    
    # 首次访问时触发自动加载
    print(f"API Key: {config.api_key}")  # 输出：自动加载API密钥... 及具体值
    print(f"Database URL: {config.database_url}")  # 输出：自动加载数据库地址... 及具体值
    
    # 第二次访问时直接使用已加载的值（不会重复加载）
    print(f"再次访问API Key: {config.api_key}")
    print(f"超时时间: {config.timeout}")

sys.exit() 

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
