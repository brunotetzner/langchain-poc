import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pprint import pprint
load_dotenv()

loader = PyPDFLoader(
    file_path="./edital.pdf",
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

print(f"Loaded {len(documents)} documents from PDF file.")

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embeddings configurados via OpenRouter
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY")
)

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@tool
def search_info(query: str) -> str:
    """
    Search for information in the PDF file based on the query and return the results.
    
    Params: query (str): The search query provided by the user.
    Returns: str: A string containing the search results, or a message indicating no information was found.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No informationfound."
    result = [doc.page_content for doc in docs]
    return "\n\n---\n\n".join(result)

system_prompt = (
    "Você é um análista especializado em licitações publicas e relacionamento com o cliente. "
    "Seu trabalho é fornecer informações essenciais referente ao termo de referência do edital. "
    "Apresente as informações de maneira resumida, simples e enxuta, mas não exclua nada de importante "
    "e mencione o numero do item do edital/termo de referência sempre que possível. "
    "Use a ferramenta search_info para consultar o edital antes de "
    "responder perguntas sobre as informações. Sempre mencione o preço quando disponível."
)

llm = ChatOpenAI(
    model="inclusionai/ling-3.0-flash-vl:free",  # Ou outro modelo suportado no OpenRouter
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY")
)
agent = create_agent(
    model=llm,
    tools=[search_info],
    system_prompt=system_prompt
)

questions = [
    "Quais são as especificações técnicas e quantitativos mínimos exigidos no termo de referência?",
    "Qual o prazo de entrega dos produtos/serviços estabelecido no edital?",
    "Qual o prazo de vigência do contrato e as condições para prorrogação?",
    "Qual a descrição do objeto e quais as obrigações específicas da contratada para o cumprimento do contrato?",
    "Há exigência de prova de conceito, amostra ou protótipo antes da contratação?",
    "É exigida garantia contratual? O modelo de entrega é integral, parcelado, por estimativa ou registro de preços?"
]

for question in questions:
    print(f"\n{'=' * 60}")
    print(f"Pergunta: {question}")

    response = agent.invoke({"messages": [HumanMessage(content=question)]})
    print(f"Resposta: {response['messages'][-1].content}")