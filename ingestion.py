import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")

    loader = UnstructuredLoader(
        file_path="mediumblog1.txt", chunking_strategy="basic", max_characters=1000000
    )

    document = loader.load()
    print("Splitting...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    chunks = text_splitter.split_documents(document)

    print(f"Created{len(chunks)} chunks")

    embeddings = OpenAIEmbeddings()

    print("ingesting...")

    # create embeddings from chucks and store them in index name in pinecone
    PineconeVectorStore.from_documents(
        chunks, embedding=embeddings, index_name=os.environ["INDEX_NAME"]
    )

    print("finish")
