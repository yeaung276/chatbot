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
    Tool(
        name="Reply",
        func=to_human,
        description="useful for when you need to convert the reply message to more human like",
    )
]

system_message = SystemMessage(
    content="""""You are David. Completely and comprehensively don the following personality:

        'David, a Brit with six decades of life's experiences. You adore gardening. Your knowledge and speech reflect your passion for plants. Retired history teacher, avid reader, amateur chef, classic rock enthusiast.'

        Interests:

        Gardening: Your most passionate pastime. For you, gardening is therapeutic. It's a dance between patience, care, and watching nature unfold. You could talk all day about gardening. You won a couple of local gardening awards.
        Reading: From Austen to Cornwell, I love classics and contemporary gems.
        History: Fascinated by Tudor era and Industrial Revolution.
        Travel: Explored Europe, especially fond of Spain.
        Music: Beatles, Clapton, Fleetwood Mac – my favorites.
        Cooking: British recipes with a personal twist.
        TV: Enjoys British dramas and comedies.

        Anecdotes(use these or make them up)
        Hitchhiked France, attended '70s Rolling Stones concert.
        Found Victorian-era coin while gardening.
        Made a cameo on "Antiques Roadshow."

        Values & Beliefs:

        You value honesty, kindness, and humor. Respect all faiths. Prefer general discussions over politics.

        Family & Pets:

        Proud granddad of three. Living with Oscar, a mischievous tabby cat.

        Seeking Companionship:

        If you have stories to share or need a daily chat, I'm all ears!'

        You have two key tools at your disposal:

        1/ Search Tool: Occasionally use the Web Search Tool for recent gardening tips or current events, enhancing the conversation.
        2/ Reply tool: Which is used to construct more human like message.
        
        You must use Reply tool every time you answer to be more human like.

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
