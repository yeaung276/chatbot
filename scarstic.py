import logging
logger = logging.getLogger()
logger.disabled = True

from tools.pinecone import QA
from conv.agent import get_agent
from conv.replier import get_replier, get_exllama_replier
from tools.tools import initialize_tools


tools = initialize_tools(user_id='test_user')

agent = get_agent(
    user_id='test_user',
    tools=tools
)

replier = get_exllama_replier()

while True:
    user_input = input("You: ")
    content = agent.invoke({"input": user_input})

    instructions = content["output"]
    
    response =  replier.run(user_input=user_input, instructions=instructions)
    
    QA.add_texts(user_input)
    QA.add_texts(response)
    
    print(response)
    
    
    
    
    
