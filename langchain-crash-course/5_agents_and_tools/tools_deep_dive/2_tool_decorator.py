# Documentation: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/

# Import necessary libraries
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
# This is where you should store your API keys (e.g., OPENAI_API_KEY)
load_dotenv()

# --- Tool Definitions using the @tool decorator ---

# Approach 1: Simple Tool without an explicit args_schema
# For tools with simple inputs, LangChain can infer the schema from type hints.
@tool
def greet_user(name: str) -> str:
    """Greets the user by name. Use this when a user wants to be greeted."""
    return f"Hello, {name}!"


# Approach 2: Tool with a single, complex argument using an explicit args_schema
# For more clarity or when you need detailed descriptions, define a Pydantic model.
class ReverseStringArgs(BaseModel):
    """Input schema for the reverse_string tool."""
    text: str = Field(description="The text to be reversed")

@tool(args_schema=ReverseStringArgs)
def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]


# Approach 3: Tool with multiple arguments using an explicit args_schema
# This is the standard approach for tools that require multiple inputs.
class ConcatenateStringsArgs(BaseModel):
    """Input schema for the concatenate_strings tool."""
    a: str = Field(description="First string to concatenate")
    b: str = Field(description="Second string to concatenate")

@tool(args_schema=ConcatenateStringsArgs)
def concatenate_strings(a: str, b: str) -> str:
    """Concatenates two strings together."""
    print(f"Tool received: a='{a}', b='{b}'")
    return a + b


# --- Agent Setup ---

# Create a list of all the tools the agent can use.
tools = [
    greet_user,
    reverse_string,
    concatenate_strings,
]

# Initialize the language model. `gpt-4o` is a strong choice for agentic tasks.
# Ensure your OPENAI_API_KEY is set in your environment variables.
llm = ChatOpenAI(model="gpt-4o")

# Pull a pre-built prompt template from LangChain Hub.
# This prompt is designed to work with OpenAI's tool-calling features.
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create the agent by combining the LLM, tools, and prompt.
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# Create the agent executor.
# This is the modern way to initialize the executor, directly passing the agent and tools.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # Set to True to see the agent's reasoning process.
    handle_parsing_errors=True, # Gracefully handle any LLM output parsing errors.
)

# --- Agent Invocation ---

# Test Case 1: Greet a user.
print("--- Invoking Agent for Greeting Task ---")
response_greet = agent_executor.invoke({"input": "Greet Alice"})
print("\n--- Final Response ---")
print(response_greet)

# Test Case 2: Reverse a string.
print("\n--- Invoking Agent for Reverse String Task ---")
response_reverse = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("\n--- Final Response ---")
print(response_reverse)

# Test Case 3: Concatenate two strings.
print("\n--- Invoking Agent for Concatenation Task ---")
response_concat = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("\n--- Final Response ---")
print(response_concat)
