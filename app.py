from typing import TypedDict, Annotated
import os
from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from retreiver_tool import neighborhood_info_tool

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [neighborhood_info_tool]
chat_with_tools = chat.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState) -> AgentState:
    return {"messages": [chat_with_tools.invoke(state["messages"])]}

graph = StateGraph(AgentState)

graph.add_node("assistant", assistant)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "assistant")
graph.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools, otherwise, provide a direct response
    tools_condition,
)
graph.add_edge("tools", "assistant")
neighborhood_assistant = graph.compile()

neighborhood_assistant.get_graph().draw_mermaid_png(
    output_file_path="neighborhood_agent_graph.png"
)

user_input = [HumanMessage(content="In which apartment does Laura Martínez live?")]
response = neighborhood_assistant.invoke({"messages": user_input})

print("🤖 Agent's Response:")
print(response['messages'][-1].content)

print("⚠️ All the messages:")
print(response['messages'])
