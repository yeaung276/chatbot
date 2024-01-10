from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from prompts.replyer import replier_prompt
from prompts.personalities import fun_and_scarstic_replyer

def get_replier(personalities=fun_and_scarstic_replyer):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    return LLMChain(
        llm=llm,
        prompt=replier_prompt.partial(personalities=personalities),
        verbose=True
    )