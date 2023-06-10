# Bring in deps
import os
from apikey import apikey
# pip install streamlit langchain openai wikipedia chromadb tiktoken
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain, SequentialChain
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
# from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from prompt import TEMPLATE_MADE_WITH_LOVE_BY_RAY

import query as q
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader




# 設定 OpenAI API Key
os.environ["OPENAI_API_KEY"] = apikey
API = apikey
# 設定 streamlit
st.set_page_config(page_title=":鹦鹉::链接: PTT-NTU版  GPT", layout="wide")  # 設定 streamlit 網頁名稱
st.title(":鹦鹉::链接: PTT-NTU版  GPT")  # 設定 streamlit 標題
# 初始化會話狀態
# 會話狀態：可以共享數據，讓應用程式記住先前的狀態
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
# 使用者輸入區塊
def get_text():
    input_text = st.text_input(
        "請輸入: ",
        st.session_state["input"],
        key="input",
        placeholder="嗨！我會根據您輸入的訊息回答問題...",
    )
    return input_text
# 開啟 new chat，並將聊天記錄存檔到 'stored_session' 中
def new_chat():
    save = []  # 創建一個空列表，用於儲存聊天記錄
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        save.append("User:" + st.session_state["past"][i])  # 存擋的格式
        save.append("Bot:" + st.session_state["generated"][i])  # 存擋的格式
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""  # 將 generated, past, input 三個變量清空，已準備開始新的聊天對話
    st.session_state.entity_memory.entity_store = {}
    st.session_state.entity_memory.buffer.clear()
# 創建 streamlit 側邊欄位
with st.sidebar.expander("選擇 Model ", expanded=False):
    MODEL = st.selectbox(
        label="Model",
        options=[
            "gpt-3.5-turbo",
            "text-davinci-003",
            "text-davinci-002",
            "code-davinci-002",
        ],
    )
st.sidebar.button("New Chat", on_click=new_chat)


# Prompt templates
title_template = PromptTemplate(input_variables=["topic"], template="告訴我有關台大{topic}的資訊")
# 創造另外多個 prompt templates
script_template = PromptTemplate(
    input_variables=["title"], template="告訴我大家對台大{title}的看法"
)
# 定義llm
llm = OpenAI(
    temperature=0.3,  # 會影響到答案的隨機性程度，因為也不是要請他寫詩或什麼，所以設低一點，回答也會比較準確
    openai_api_key=API,
    model_name=MODEL,
    max_tokens=1500,
    verbose=True,
)
if "entity_memory" not in st.session_state:
    st.session_state.entity_memory = ConversationEntityMemory(llm=llm)
    
Conversation = ConversationChain(
    llm=llm,
    prompt=TEMPLATE_MADE_WITH_LOVE_BY_RAY,
    memory=st.session_state.entity_memory,
    verbose=True,
)


# 使用者輸入
user_input = get_text()
if user_input:
    r = q.multiple_filter(user_input)
    q.data_cleaner(r)
    input_file = 'NTU_library.csv'  # 輸入的CSV檔案名稱
    output_file = 'NTU_library.txt'  # 輸出的TXT檔案名稱

    q.merge_page_content(input_file, output_file)
    loader = TextLoader('NTU_library.txt')
    data = loader.load()
    chain = load_summarize_chain(llm, chain_type='map_reduce', verbose=True)
    output2 = chain.run(data)

    # title = title_chain.run(topic = user_input)
    # script = script_chain.run(title=title)
    # chain = chain.run

    output = Conversation.run(input=user_input)
    output()
    st.session_state.past.append(user_input)
    #st.session_state.generated.append(output)
    st.session_state.generated.append(output)



    # st.write(title)
    # st.write(script)
# 讓使用者可以下載聊天紀錄
download_str = []
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.info(st.session_state["past"][i], icon=":眼睛:")
        st.success(st.session_state["generated"][i], icon=":机器人脸:")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    joined_download_str = "\n".join(download_str)
    if joined_download_str:
        st.download_button("Download", joined_download_str)
# 將儲存之對話記錄存到 sidebar
for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label=f"Conversation-Session:{i}"):
        st.write(sublist)
# 讓使用者可以刪除所有對話
if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session