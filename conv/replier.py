from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from prompts.replyer import replier_prompt
from prompts.personalities import fun_and_scarstic_replyer

from conv.exllama_langchain import Exllama


def get_replier(personalities=fun_and_scarstic_replyer):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    return LLMChain(
        llm=llm,
        prompt=replier_prompt.partial(personalities=personalities),
        verbose=True,
    )


def get_exllama_replier(personalities=fun_and_scarstic_replyer):
    llm = Exllama(
        model_path="Pygmalion-13B-SuperHOT-8K-GPTQ",
        # lora_path = 'path'
        temperature=0.1,
        typical=0.5,
        beams=2,
        beam_length=40,
        stop_sequences=[],
        verbose=True,
        max_seq_len=4048,
        fused_attn=False,
        token_repetition_penalty_max = 1.15,
        # token_repetition_penalty_sustain = 256,
        alpha_value=1.0,
        compress_pos_emb=2.0,
        # set_auto_map = "3, 2" #Gpu split, this will split 3gigs/2gigs
    )
    return LLMChain(
        llm=llm,
        prompt=replier_prompt.partial(personalities=personalities),
        verbose=True,
    )
