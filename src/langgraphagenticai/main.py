from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from src.langgraphagenticai.config import GROQ_API_KEY, GROQ_MODEL
from src.langgraphagenticai.states.states import State
from src.langgraphagenticai.nodes.node import Bot1
from src.langgraphagenticai.utils import node_summary


def get_llm():
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)


def route_after_router(state: State) -> str:
    """Route based on router decision."""
    action = state.get("action", "STOP")
    if action == "TAVILY_SEARCH":
        return "tavily"
    if action == "VECTOR_STORE":
        return "vector_store"
    if action == "SEND_EMAIL":
        return "send_email"
    return "stop"


def build_graph():
    llm = get_llm()
    bot = Bot1(llm)

    graph = StateGraph(State)

    graph.add_node("router", bot.routing)
    graph.add_node("tavily", bot.tavily_search_node)
    graph.add_node("vector_store", bot.retrieve)
    graph.add_node("grade", bot.grade_documents)
    graph.add_node("generate", bot.generate)
    graph.add_node("send_email", bot.send_email_placeholder)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_after_router, {
        "tavily": "tavily",
        "vector_store": "vector_store",
        "send_email": "send_email",
        "stop": "generate"
    })
    graph.add_edge("tavily", "generate")
    graph.add_edge("vector_store", "grade")
    graph.add_edge("grade", "generate")
    graph.add_edge("send_email", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def create_initial_state(user_input: str, messages: list = None) -> dict:
    msgs = messages or []
    new_msgs = msgs + [{"role": "user", "content": user_input}]
    return {
        "input": user_input,
        "messages": new_msgs,
        "output": "",
        "decision": "",
        "action": "",
        "category": "",
        "relevent": "",
        "documents": []
    }


def run_with_streaming(user_input: str, messages: list = None) -> list:
    """Run graph, stream the answer, print node summary once at end.
    Returns updated messages (history + user + assistant) for next turn."""
    g = build_graph()
    state = create_initial_state(user_input, messages)
    summaries = []
    result = {}

    for chunk in g.stream(state, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            summaries.append(f"{node_name} -> {node_summary(node_name, node_output)}")
            if node_name == "generate":
                result = node_output

    print("\n")
    if summaries:
        print("[Flow]", " | ".join(summaries))

    # Append assistant reply to history for next turn
    new_msgs = list(state.get("messages", []))
    if result and result.get("output"):
        new_msgs.append({"role": "assistant", "content": result["output"]})
    return new_msgs
