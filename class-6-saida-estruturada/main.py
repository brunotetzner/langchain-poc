from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pprint import pprint

load_dotenv()

class PersonalInfo(BaseModel):
    name: str = Field(..., description="The name of the user")
    age: int = Field(..., description="The age of the user")
    profession: str = Field(..., description="The profession of the user")

model = ChatOpenRouter(
    model="deepseek/deepseek-v4-flash",
    temperature=0,
)

agent = create_agent(
    model,
    checkpointer=InMemorySaver(),
    response_format=PersonalInfo
)

response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="Extraia os dados: Clayson lime, 23 anos, desenvolvedor de software."),
        ]
    }
)

pprint(response)

personal_info = response["structured_response"]