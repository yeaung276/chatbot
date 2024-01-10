from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType, Tool

from prompts.agent import fun_and_scarstic

def get_agent(user_id: str, tools: list[Tool], personalities=fun_and_scarstic, verbose=True):
    agent_kwargs = {
        "system_message": personalities, 
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")]
        }

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    memory = ConversationBufferMemory(
        memory_key="memory", 
        return_messages=True, 
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"), 
        max_token_limit=2000
        )

    return initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=verbose,
        agent_kwargs=agent_kwargs,
        memory=memory,
        max_iterations=3,
        early_stopping_method="generate",
    )