import requests
from dotenv import load_dotenv

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.prebuilt import create_react_agent

load_dotenv()

OLLAMA_URL = "http://localhost:11434"

# Step 1: Initialize Ollama Models
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    base_url=OLLAMA_URL
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_URL
)

# Step 2: Tools Setup
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'
    response = requests.get(url)
    return str(response.json())

tools = [search_tool, get_weather_data]

# Step 3: Create ReAct Agent (LangGraph Prebuilt)
agent_executor = create_react_agent(
    model=llm,
    tools=tools
)

# Step 4: Invoke Agent
if __name__ == "__main__":
    query = "What is the current temp of gurgaon"
    
    response = agent_executor.invoke({"messages": [("user", query)]})
    
    print("\n--- Final Answer ---")
    print(response["messages"][-1].content)