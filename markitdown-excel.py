from markitdown import MarkItDown

md = MarkItDown(enable_plugins=False) # Set to True to enable plugins
result = md.convert("/Users/bao.li001/codebuddy/LangchainProj/credit_amount_debug.xls")
print(result.text_content)
