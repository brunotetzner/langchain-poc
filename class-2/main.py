from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# a interação não necessáriamente precisa ter uma unica mensagem(one shot)
# existem diferentes tipos de mensagem
    # mensagem de humano
    # mensagem de sistema - instruções para o modelo de compo se comportar e respoonder


load_dotenv()

model = ChatOpenRouter(
    model="deepseek/deepseek-v4-flash",
    temperature=0
)

messages = [
    SystemMessage(content="Você é um assistente que responde apenas em forma de haiku"),
    HumanMessage(content="Como funciona o langchain?")
]


for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)

