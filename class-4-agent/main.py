from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import  HumanMessage, AIMessage
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool
from pprint import pprint
load_dotenv()

@tool
def calculate_bmi(weight: float, height: float) -> str:
    """
    Calcula o IMC (Índice de Massa Corporal) com base no peso e altura fornecidos
    
    Args:
        weight (float): Peso em quilogramas
        height (float): Altura em metros

    Returns:
        str: Uma string formatada com o valor do IMC calculado
    ."""
    bmi = weight / (height ** 2)

    print(f"O seu IMC é: {bmi:.2f}")
    return f"O seu IMC é: {bmi:.2f}"


model = ChatOpenRouter(
    model="deepseek/deepseek-v4-flash",
    temperature=0,
)
agent = create_agent(
    model,
    tools=[calculate_bmi],
    system_prompt="Você é um assistente que calcula o IMC (Índice de Massa Corporal) com base no peso e altura fornecidos pelo usuário. Quando o usuário fornecer seu peso em quilogramas e altura em metros, você deve calcular o IMC usando a fórmula: IMC = peso / (altura ** 2). Em seguida, você deve retornar o valor do IMC calculado ao usuário."
)

messages = [
    HumanMessage(content="Qual é o meu IMC se eu pesar 70kg e medir 1,75m?")
]

# response = agent.invoke({"messages": messages})

# print(response["messages"][-1].content)
# pprint(response)

for chunk in agent.stream({"messages": messages}, stream_mode="values"):
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"Human: {latest_message.content}")
        if isinstance(latest_message, AIMessage):
            print(f"AI: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Tool Calls: {[tc["name"] for tc in latest_message.tool_calls]}")
