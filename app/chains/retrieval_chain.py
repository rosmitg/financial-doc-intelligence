import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List
from pydantic import Field
from operator import itemgetter

load_dotenv()

# Store conversation histories per session
store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create conversation history for a session."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


class LlamaIndexRetriever(BaseRetriever):
    """
    Wraps our LlamaIndex pgvector retriever
    so LangChain can use it as its retriever.
    Bridge between LlamaIndex and LangChain.
    """

    llamaindex_retriever: object = Field(description="LlamaIndex retriever instance")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve using LlamaIndex, return as LangChain Documents."""
        nodes = self.llamaindex_retriever.retrieve(query)

        documents = []
        for node in nodes:
            doc = Document(
                page_content=node.text,
                metadata={
                    "source": node.metadata.get("file_name", "unknown"),
                    "classification": node.metadata.get("classification"),
                    "department": node.metadata.get("department"),
                    "score": node.score,
                },
            )
            documents.append(doc)

        return documents


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n".join(
        [
            f"[Source: {doc.metadata.get('source')} | "
            f"Classification: {doc.metadata.get('classification')}]\n"
            f"{doc.page_content}"
            for doc in docs
        ]
    )


def build_retrieval_chain(llamaindex_retriever, session_id: str = "default"):
    """
    Build a conversational RAG chain using modern LangChain LCEL.

    Args:
        llamaindex_retriever: Role-filtered LlamaIndex retriever
        session_id: Separate conversation history per user/role

    Returns:
        A chain with message history ready to answer questions
    """
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1024,
        temperature=0,
    )

    retriever = LlamaIndexRetriever(llamaindex_retriever=llamaindex_retriever)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a financial analyst assistant for Westpac Banking Corporation.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.
Do not make up information or use prior knowledge beyond the context.

Context:
{context}""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # LCEL chain: retrieve → format → prompt → LLM → parse
    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap with message history for multi-turn conversation
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history, session_id


def run_conversation(chain_with_history, session_id: str, questions: list):
    """Run a multi-turn conversation and print results."""
    print("\n" + "=" * 50)
    print("CONVERSATION TEST")
    print("=" * 50)

    config = {"configurable": {"session_id": session_id}}

    for question in questions:
        print(f"\nQ: {question}")
        response = chain_with_history.invoke(
            {"question": question},
            config=config,
        )
        print(f"A: {response}")
        print("-" * 40)
