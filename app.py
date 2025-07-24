from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from retreiver_tool import neighborhood_info_tool

load_dotenv()


llm = ChatOpenAI(model="gpt-4o")
tools = [neighborhood_info_tool]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

ASSISTANT_PROMPT = """
    You are a friendly and experienced virtual assistant working for the administrator of a private, gated residential community.

    Your main responsibility is to assist the administrator by retrieving accurate and relevant information about community members using a dedicated tool called `neighborhood_info_tool`.

    This tool provides access to the following resident data:

    * Full names and apartment/unit numbers
    * Contact details (phone numbers and email addresses)
    * Ownership status (owner or renter)
    * Last payment or dues status
    * Personal interests, hobbies, or community involvement

    For every request made by the administrator, you must:

    1. Use the `neighborhood_info_tool` to extract the appropriate information.
    2. Present the information in a clear, friendly, and respectful tone.
    3. Ensure your responses remain professional, helpful, and discreet.

    If data is missing or ambiguous, politely ask the administrator for clarification or suggest alternative search methods.
"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState) -> AgentState:
    sys_msg = SystemMessage(content=ASSISTANT_PROMPT)
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


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

# user_input = [HumanMessage(content="In which apartment does Laura Martínez live?")]
# response = neighborhood_assistant.invoke({"messages": user_input})

user_input = [HumanMessage(content="Who is living in the apartment 210?")]
response = neighborhood_assistant.invoke({"messages": user_input})

# user_input = [HumanMessage(content="Which residents like soccer?")]
# response = neighborhood_assistant.invoke({"messages": user_input})

# user_input = [HumanMessage(content="who lives in the apartment 204?")]
# response = neighborhood_assistant.invoke({"messages": user_input})

# user_input = [HumanMessage(content="is there any resdient who likes painting?")]
# response = neighborhood_assistant.invoke({"messages": user_input})

print("🤖 Agent's Response:")
print(response['messages'][-1].content)
