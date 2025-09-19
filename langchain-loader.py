import bs4
from langchain_community.document_loaders import WebBaseLoader


import asyncio

page_url = "https://docs.naiveadmin.com/guide/"


async def test():
    loader = WebBaseLoader(web_paths=[page_url])
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    assert len(docs) == 1
    doc = docs[0]
    print(f"{doc.metadata}\n")
    print(doc.page_content[:5000].strip())


async def test2():
    loader = WebBaseLoader(
        web_paths=[page_url],
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(class_="language-bash"),
        },
        bs_get_text_kwargs={"separator": " | ", "strip": True},
    )

    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    assert len(docs) == 1
    doc = docs[0]
    print(f"{doc.metadata}\n")
    print(doc.page_content[:5000].strip())
    
if __name__ == "__main__":
    asyncio.run(test2())