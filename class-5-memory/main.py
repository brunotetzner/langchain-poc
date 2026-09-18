from langchain_openrouter import ChatOpenRouter

from langchain.agents import create_agent

from langchain.messages import HumanMessage

from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv



load_dotenv()



model = ChatOpenRouter(

    model="deepseek/deepseek-v4-flash",

    temperature=0,

)

agent = create_agent(

    model,

    checkpointer=InMemorySaver()

)



config = {"configurable": { "thread_id": "user-123" }}



response = agent.invoke({"messages":[HumanMessage(content="Olá, meu nome é Cleyson")]}, config)

print(response["messages"][1].content)



response = agent.invoke({"messages":[HumanMessage(content="Qual é o meu nome?")]}, config)

print(response["messages"][-1].content)
