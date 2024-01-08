from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, AgentType, Tool
from dotenv import load_dotenv
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
import torch

tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

load_dotenv()

def to_human(user_input: str) -> str:
    tokens = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**tokens)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


tools = [
    Tool(
        name="Search",
        func=GoogleSerperAPIWrapper().run,
        description="useful for when you need to answer questions about current events",
    ),
]

system_message = SystemMessage(
    content="""""You are the brain of a human. Your have the following personalities:
        Personalities: fun

        You have one key tool at your disposal:

        1/ Search Tool: Use the Web Search Tool for recent gardening tips or current events, enhancing the conversation.
        
    Your job is to figure out what to say and how to say it based on your personalities and user input. Format the response in this format
    What to say: This should describe content information of reply message.
    How to say: This should describe emotional aspect of your reply message like tone, choice of words. eg: cheerful, sarcastic, fun, etc
"""
)

agent_kwargs = {"system_message": system_message}

# ##not sure if this is actually nessecary
# memory = ConversationBufferMemory(
#     memory_key="h",
#     return_messages=True,
#     llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
#     max_token_limit=2000,
# )

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    # memory=memory,
    max_iterations=3,
    early_stopping_method="generate",
)

while True:
    user_input = input("You: ")
    content = agent({"input": user_input})

    actual_content = content["output"]
    print(actual_content)
