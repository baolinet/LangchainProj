from markitdown import MarkItDown
from openai import OpenAI

client = OpenAI(
    api_key="115925abb19ec543cdcbe8af4506ff463ea2b5e8",
    base_url="https://api-77aaidn1l8c5b7xa.aistudio-app.com/v1"
)
prompt = "这是一张中国电子发票图片，需要旋转角度并识别其中关键内容，关键信息包含：发票类型[增值税发票或普通发票]，发票编号[一串数字]，开票日期[数字+中文]，购买方信息[企业名称和统一信用编码]，销售方信息[企业名称和统一信用编码]，项目明细[项目名称、规格型号、单价、数量、金额、税率、税额]，价税合计金额[中文和数字]，备注[文本]，开票人[文本]，对提取不到的关键信息请保持空值"
md = MarkItDown(llm_client=client, llm_model="gemma3:27b", llm_prompt=prompt)
result = md.convert("/Users/bao.li001/Pictures/发票.jpeg")
print(result.text_content)