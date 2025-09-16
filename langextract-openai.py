# pip install langextract[openai]

import langextract as lx
import textwrap

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""\
    按外观顺序提取角色、情感和关系。
    使用精确的文本进行提取。不要意译或重叠实体。
    为每个实体提供有意义的属性，以添加上下文。""")

# 2. Provide a high-quality example to guide the model
examples = [
    lx.data.ExampleData(
        text="罗密欧，但愿！那窗外的光是从何而来？那是东方，朱丽叶就是太阳。",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="罗密欧",
                attributes={"emotional_state": "想知道"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="但愿！",
                attributes={"feeling": "温柔的敬畏"}
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="朱丽叶就是太阳",
                attributes={"type": "比喻"}
            ),
        ]
    )
]

# The input text to be processed
input_text = "朱丽叶夫人渴望地凝视着星星，她的心为罗密欧而痛心。"

from langextract import factory

config = factory.ModelConfig(
    provider="OpenAILanguageModel",
    provider_kwargs={
        "api_key":"115925abb19ec543cdcbe8af4506ff463ea2b5e8",
        "base_url":"https://api-77aaidn1l8c5b7xa.aistudio-app.com/v1",
        "temperature": 0.6,
    },
    model_id="gemma3:27b",
)

model = factory.create_model(config)

# Run the extraction
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model=model,
)

for extraction in result.extractions:
    print(extraction.extraction_index, extraction.extraction_class, extraction.extraction_text, extraction.attributes)

# [Extraction(extraction_class='character', extraction_text='朱丽叶夫人', char_interval=None, alignment_status=None, extraction_index=1, group_index=0, description=None, attributes={'emotional_state': '渴望', 'action': '凝视着星星'}), Extraction(extraction_class='emotion', extraction_text='心为罗密欧而痛心', char_interval=None, alignment_status=None, extraction_index=2, group_index=1, description=None, attributes={'feeling': '悲伤', 'target': '罗密欧'}), Extraction(extraction_class='character', extraction_text='罗密欧', char_interval=None, alignment_status=None, extraction_index=3, group_index=2, description=None, attributes={'role': '被思念的人'})]