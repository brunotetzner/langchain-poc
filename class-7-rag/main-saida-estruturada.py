import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

class ResponsePayload(BaseModel):
    to: str = Field(..., description="The name of the user")


loader = CSVLoader(
    file_path="./products.csv",
    encoding="utf-8"
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

print(f"Loaded {len(documents)} documents from CSV file.")

# Embeddings configurados via OpenRouter
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY")
)

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@tool
def search_products(query: str) -> str:
    """
    Search for products in the CSV file based on the query and return the results.
    
    Params: query (str): The search query provided by the user.
    Returns: str: A string containing the search results, or a message indicating no products were found.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No products found."
    result = [doc.page_content for doc in docs]
    return "\n\n---\n\n".join(result)

system_prompt = (
    "Você é um assistente de vendas especializado. "
    "Use a ferramenta search_products para consultar o catálogo antes de "
    "responder perguntas sobre os produtos. Sempre mencione o preço quando disponível."
)

llm = ChatOpenAI(
    model="deepseek/deepseek-v4-flash",  # Ou outro modelo suportado no OpenRouter
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY")
)
agent = create_agent(
    model=llm,
    tools=[search_products],
    system_prompt=system_prompt,
    response_format=ResponsePayload
)

questions = [
    "Quais produtos você tem para melhorar a ergonomia do home office?",
    "Estou montando um setup gamer, tem algo para recomendar?",
    "Preciso de periféricos sem fio, quais opções vocês tem?",
    "Qual o produto mais barato do catálogo?"
]

for question in questions:
    print(f"\n{'=' * 60}")
    print(f"Pergunta: {question}")

    response = agent.invoke({"messages": [HumanMessage(content=question)]})
    print(f"Resposta: {response['messages'][-1].content}")