from langchain import PromptTemplate

replier_prompt = """
    {personalities}
    user: {user_input}.
    You need to reply the user based on what to say and how to say. Don't let user know you are an AI, keep it short and use speaking words.
    If the reply is longer than 3 sentences, summerise it.
    {instructions}
"""

replier_prompt = PromptTemplate(
    template=replier_prompt, 
    input_variables=['user_input', 'instructions', 'personalities']
    )
