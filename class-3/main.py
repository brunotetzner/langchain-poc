from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import  HumanMessage
from dotenv import load_dotenv
from langchain.tools import tool

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
    temperature=0
)
models_with_tools = model.bind_tools([calculate_bmi])

messages = [
    HumanMessage(content="Qual é o meu IMC se eu pesar 70kg e medir 1,75m?")
]


response = models_with_tools.invoke(messages)

messages.append(response)
for toll_call in response.tool_calls:
    result = calculate_bmi.invoke(toll_call)
    messages.append(result)

final = models_with_tools.invoke(messages)