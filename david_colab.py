import logging
logger = logging.getLogger()
logger.disabled = True

# from tools.pinecone import QA
from conv.agent import get_agent
from conv.replier import get_exllama_replier
from tools.tools import initialize_tools
from prompts.agent import david
from prompts.personalities import david_replier


tools = initialize_tools(user_id='test_user')

agent = get_agent(
    user_id='test_user',
    tools=tools,
    personalities=david
)

replier = get_exllama_replier(
    personalities=david_replier
)

with open('conv_record.txt', 'w+') as f:
    for user_input in convs:
        content = agent.invoke({"input": user_input})

        instructions = content["output"]
        
        response =  replier.run(user_input=user_input, instructions=instructions)
        
        # QA.add_texts(user_input)
        # QA.add_texts(response)
        
        print(f'AI: {response}')
        f.write(f'User: {user_input}\n')
        f.write(f"AI: {response}\n\n")