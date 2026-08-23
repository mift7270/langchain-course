import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()

print("Initializing components...")

# create embedding object: its job during retrival will be take the user's question and convert into a vector
embeddings = OpenAIEmbeddings()

# object used to create the final answer through LLM
llm = ChatOpenAI()


# This is saying: Langchain, connect me to this pinecone index, and use this embedding model when I need to convert text queries into vectors
vectostore = PineconeVectorStore(
    embedding=embeddings, index_name=os.environ["INDEX_NAME"]
)

# I want to use this vector store as a document retrieval system and Return the top 3 most relevant chunks.
retriever = vectostore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """ Answer the question based only on the following context:

  {context}

  question :{question}

  provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


# =======================================
# Option 1: Use implentation WITHOUT LCEL
# ======================================


def retrival_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a resposne
    Limitations:
    -Manual step by step execution
    -No built-in streaming support
    -No async support without addtional code
    -Harder to compose with other chains
    -More verbose and error prone
    """
    # step1:Retreive revalnt docuemnts
    docs = retriever.invoke(query)

    # step2: Format documents into a context string
    context = format_docs(docs)

    # step 3: Format the promoit withteh context and question

    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)

    return response.content


# ======================================================================
# Option 2: With LCEL (Langchain expression languate (Better Approach)
# =====================================================================


def create_retrival_chain_with_lcel():
    """
    Create a retrival chain usin LCEL (Langchain expressoni Language)
    Return a chain that can be invoved with {"question":"..."}

    Advantes over non-LCEL appreach:
    -Declarative and composable: Easy to chain operations with pipe operator (|)
    -Built-in streaming: Chain.stream() works out for hte box
    -Build-in async: chain.ainvoke() and chain.astream() available
    -Batch processing: chain.batch() for multiple inputs
    -Type safety: Better integration with Langchain's type system
    -Less code: More concise and readable
    -Reusable: Chain can be saved, shared, and composed with other chains
    -Better debugging: Langchian provides better observability tools
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    # query

    query = "what is picone in machine learning"

    # =======================================
    # Option 0: Raw invocation without RAG
    # ======================================

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("\n" + "=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # =======================================
    # Option 1: Use implentation WITHOUT LCEL
    # ======================================

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: WITHOUT LCEL")
    print("\n" + "=" * 70)
    result_without_lcel = retrival_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # =======================================
    # Option21: Use implentation WITH LCEL (Better Appreach)
    # ======================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: WITH LCEL - Better Approach")
    print("\n" + "=" * 70)
    print("\n" + "=" * 70)

    chain_with_lcel = create_retrival_chain_with_lcel()
    result_without_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer")
    print(result_without_lcel)
