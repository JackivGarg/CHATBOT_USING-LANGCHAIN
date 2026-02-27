import sys
from src.langgraphagenticai.states.states import State, Route1, GradeDocument
from src.langgraphagenticai.prompts import router_template, document_grader_template, generate_template
from src.langgraphagenticai.utils import format_history
from src.tools.tools import load_vector_store

_tavily_tool = None


def _get_tavily():
    global _tavily_tool
    if _tavily_tool is None:
        from langchain_community.tools.tavily_search import TavilySearchResults
        _tavily_tool = TavilySearchResults(max_results=5, search_depth="advanced")
    return _tavily_tool


class Bot1:
    def __init__(self, model):
        self.llm = model

    def routing(self, state: State):
        router_chain = router_template | self.llm.with_structured_output(Route1)

        msgs = state.get("messages", [])   #ifwe use state[messages] and it isnt present it would give error -> now if its emplty it would return []
        history_str = format_history(msgs)
        result = router_chain.invoke({
            "user_input": state["input"],
            "history": history_str or "(none)",
            "context": state.get("output", "")
        })

        state["action"] = result.action
        state["category"] = result.category
        return state

    def tavily_search_node(self, state: State) -> dict:
        query = state["input"]
        results = _get_tavily().invoke(query)
        context = "\n".join(
            f"- {r.get('title','')}\n  {r.get('content','')}\n  Source: {r.get('url','')}"
            for r in results
        )
        return {"output": context}
    



    def retrieve(self, state: State) -> dict:
        """
        Retrieve documents from the correct vector store
        based on router category. Stores documents for grading.
        """
        question = state["input"]
        category = state.get("category")

        if not category:
            return {"output": "No category found for vector retrieval.", "documents": []}

        vectorstore = load_vector_store(category)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        documents = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in documents)

        return {"output": context, "documents": documents}



    def grade_documents(self, state: State) -> dict:
        """
        Grade retrieved documents for relevance.
        If no docs relevant -> relevent=no, output="" (generate will use LLM knowledge).
        """
        question = state["input"]
        documents = state.get("documents") or []

        if not documents:
            return {"relevent": "no", "output": ""}

        relevant_docs = []
        grader_chain = document_grader_template | self.llm.with_structured_output(GradeDocument)

        for doc in documents:
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            result = grader_chain.invoke({"question": question, "document": content})
            if result.relevance == "yes":
                relevant_docs.append(doc)

        if not relevant_docs:
            return {"relevent": "no", "output": ""}

        context = "\n\n".join(
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in relevant_docs
        )
        return {"relevent": "yes", "output": context}

    def generate(self, state: State) -> dict:
        """Generate final answer from context. Streams tokens as they arrive."""
        context = state.get("output") or ""
        user_input = state.get("input") or ""
        history_str = format_history(state.get("messages", []))

        chain = generate_template | self.llm
        answer_parts = []
        for chunk in chain.stream({"context": context, "history": history_str or "(none)", "user_input": user_input}):
            txt = chunk.content if hasattr(chunk, "content") else str(chunk)
            if txt:
                sys.stdout.buffer.write(txt.encode("utf-8", errors="replace"))
                sys.stdout.flush()
                answer_parts.append(txt)
        answer = "".join(answer_parts)

        return {
            "output": answer,
            "messages": [{"role": "assistant", "content": answer}]
        }

    def send_email_placeholder(self, state: State) -> dict:
        """Placeholder for Gmail - will add later."""
        state["output"] = ""
        return state


    