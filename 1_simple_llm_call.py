from dotenv import load_dotenv

# Step 1: Load environment variables from .env file
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatOllama(model="qwen2.5:3b")
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of India?"})
print(result)
