from langchain_openrouter import ChatOpenRouter

model = ChatOpenRouter(
    model="deepseek/deepseek-v4-flash-0731",
    temperature=0
)

response = model.invoke("Explique o que é uma arquitetura de microsserviços.")

print(response.content)