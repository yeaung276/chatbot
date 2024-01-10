import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.tools import StructuredTool
from langchain.chat_models import ChatOpenAI

import variables
from prompts.retriever import retriever_prompt

index = pinecone.Index(variables.index_name)
pinecone.init(api_key=variables.pinecone_api_key, environment=variables.pinecone_enviroment)
embeddings = OpenAIEmbeddings(max_retries = 3)
vectorstore = Pinecone.from_existing_index(index_name=variables.index_name, embedding=embeddings, text_key= "summary")

class QA:
    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        
    def __call__(self, question: str) -> str:
        retriever = vectorstore.as_retriever(
            search_kwargs={
                'filter': {'userID':self.chat_id},
                'k':5
            })

        chain_type_kwargs = {"prompt": retriever_prompt}

        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

        qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)


        ans = qa_chain.run({"query": question})

        return ans
    
    
    @staticmethod
    def add_texts(text: str):
        vectorstore.add_texts(text)

def get_history_tool(chat_id: str):
    retriever = QA(chat_id)
    return StructuredTool.from_function(
        name="Chat_History",
        description="Useful when wanting to recall and contextualize previous conversations with a user, enhancing current interactions by referencing past topics or challenges.",
        func=lambda question: retriever(question)
    )
