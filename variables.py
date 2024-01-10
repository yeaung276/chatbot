import os
from dotenv import load_dotenv

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_enviroment = os.getenv("PINECONE_ENVIRONMENT")
openai_api_key = os.getenv("OPENAI_API_KEY")
length_chat_history = os.getenv("LENGTH_CHAT_HISTORY")
chat_history_table= os.getenv("CHAT_HISTORY_TABLE")
index_name = os.getenv("PINECONE_INDEX_NAME")