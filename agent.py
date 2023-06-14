from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper

from langchain import LLMChain

from agent_prompt_template import prompt_with_history, output_parser, tools
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser



llm = OpenAI(openai_api_key="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",temperature=0)

llm_chain = LLMChain(llm=llm, prompt=prompt_with_history) #NEW_ADDED




tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
) #NEW_ADDED

from langchain.memory import ConversationBufferWindowMemory
memory=ConversationBufferWindowMemory(k=2)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory) #NEW_ADDED
agent_executor.run("台大學生對女生讀資工系的看法如何？") #NEW_ADDED





#agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
#agent.run("請問在2021年時，台大學生對女生讀資工系的看法如何？")