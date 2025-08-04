# Docs: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/

# Import necessary libraries
import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# --- Tool Function Definitions ---
# These are the Python functions that your tools will execute.

def greet_user(name: str) -> str:
    """Greets the user by name."""
    return f"Hello, {name}!"

def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]

def concatenate_strings(a: str, b: str) -> str:
    """Concatenates two strings."""
    return a + b

# --- Tool Creation using StructuredTool ---
# Using StructuredTool.from_function is a consistent way to create tools.
# It automatically infers the schema from the function's type hints.

tools = [
    # Tool for a function with a single argument.
    StructuredTool.from_function(
        func=greet_user,
        name="GreetUser",
        description="Greets the user by name. Use this when a user asks to be greeted.",
    ),
    # Tool for another function with a single argument.
    StructuredTool.from_function(
        func=reverse_string,
        name="ReverseString",
        description="Reverses a given string. Useful for text manipulation.",
    ),
    # Tool for a function with multiple arguments.
    # The schema is inferred from the function signature and docstrings.
    StructuredTool.from_function(
        func=concatenate_strings,
        name="ConcatenateStrings",
        description="Concatenates two strings together. Use this when you need to join text.",
    ),
]

# --- Custom Prompt Template ---
# Instead of pulling from the hub, we define our own prompt here.
# This gives you full control over the agent's instructions and personality.

# Note the three key variables: `input`, `chat_history`, and `agent_scratchpad`.
# These are required for the agent to work correctly.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and friendly assistant. You have access to a variety of tools to help users with their tasks. "
            "When a user asks for something, select the best tool to accomplish their goal and respond with the result. "
            "Always be polite and provide clear, concise answers."
        ),
        # `MessagesPlaceholder` is used to pass the conversation history.
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        # `agent_scratchpad` is where the agent stores its intermediate steps (tool calls and observations).
        # This is crucial for the agent's reasoning process.
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# --- Agent Setup ---

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o")

# Create the agent by combining the LLM, tools, and our custom prompt.
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# Create the agent executor.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # See the agent's thought process
    handle_parsing_errors=True,
)

# --- Agent Invocation ---

# Test Case 1: Greet a user
print("--- Invoking Agent for Greeting Task ---")
response_greet = agent_executor.invoke({"input": "Greet Alice"})
print("\n--- Final Response ---")
print(response_greet)

# Test Case 2: Reverse a string
print("\n--- Invoking Agent for Reverse String Task ---")
response_reverse = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("\n--- Final Response ---")
print(response_reverse)

# Test Case 3: Concatenate two strings
print("\n--- Invoking Agent for Concatenation Task ---")
response_concat = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("\n--- Final Response ---")
print(response_concat)
