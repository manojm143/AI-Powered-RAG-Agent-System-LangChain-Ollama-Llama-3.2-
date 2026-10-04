from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool

from rag import retriever


# -----------------------------
# 1. Calculator tool
# -----------------------------

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    return str(eval(expression))


# -----------------------------
# 2. RAG tool
# -----------------------------

@tool
def rag_tool(query: str) -> str:
    """Search the dargon fruit information regarding paper."""

    results = retriever.invoke(query)

    return "\n\n".join(
        doc.page_content
        for doc in results
    )


# -----------------------------
# 3. Llama
# -----------------------------

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)


# -----------------------------
# 4. Create agent
# -----------------------------

agent = create_agent(
    model=llm,
    tools=[
        calculator,
        rag_tool
    ]
)


# -----------------------------
# 5. Chat
# -----------------------------

while True:

    question = input(
        "\nAsk a question (type exit to stop): "
    )

    if question.lower() == "exit":
        break

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    print("\nAnswer:")
    print(result["messages"][-1].content)