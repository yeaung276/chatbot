from langchain import PromptTemplate

prompt_template = """You are being asked a question. It is part of a function called by a langchain agent to help with a question.
        Answer the question based on the context provided from past conversations. You have to tell if anything related to the question
        being asked has been discussed in the past. Tell whatever related was discussed. If nothing related to the query was discussed, simply say so.
        Before your answer, include the context provided to you in this format: Context:.....    Answer:.....

        Context: {context}
        Question: {question}
        Answer:"""

retriever_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)