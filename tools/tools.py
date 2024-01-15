from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool

# from tools.pinecone import get_history_tool


def initialize_tools(user_id: str):
    return [
        Tool(
            name="Search",
            func=GoogleSerperAPIWrapper().run,
            description="useful for when you need to answer questions about current events",
        ),
        # get_history_tool(user_id)
    ]