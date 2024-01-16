import logging
logger = logging.getLogger()
logger.disabled = True

# from tools.pinecone import QA
from conv.agent import get_agent
from conv.replier import get_replier
from tools.tools import initialize_tools
from prompts.agent import david
from prompts.personalities import david_replier


tools = initialize_tools(user_id='test_user')

agent = get_agent(
    user_id='test_user',
    tools=tools,
    personalities=david
)

replier = get_replier(
    personalities=david_replier
)

for user_input in ['hi, what is your name', 'tell me about your day', 
'how was the weather there', 'share me your life lesson', 
'what is tranding in today internet', 'have you killed someone',
'do you have a wife?'
]:
    content = agent.invoke({"input": user_input})

    instructions = content["output"]
    
    response =  replier.run(user_input=user_input, instructions=instructions)
    
    # QA.add_texts(user_input)
    # QA.add_texts(response)
    
    print(f'AI: {response}')