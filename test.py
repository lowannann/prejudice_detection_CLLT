import query as q
from langchain.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint

#r = q.multiple_filter("資工系")
#q.data_cleaner(r)




file_path='NTU_library.json'
data = json.loads(Path(file_path).read_text())
pprint(data)

#調整 json 格式
formatted_data = [{'post_id': item['post_id'], 'page_content': item['page_content']} for item in data]
with open("NTU_library.json", "w") as json_file:
    json.dump(formatted_data, json_file)
#印出調整後的 json 內容
file_path='NTU_library.json'
new_data = json.loads(Path(file_path).read_text())
pprint(new_data)


loader = JSONLoader(
    file_path='NTU_library.json',
    jq_schema='[.page_content]')

document = loader.load()
pprint(document)

