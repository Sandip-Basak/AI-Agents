# Import necessary libraries
import os
from typing import Type, Any

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Pydantic models for tool arguments
# These models define the expected input schema for your custom tools.
# Using Pydantic ensures that the inputs to your tools are validated.

class SimpleSearchInput(BaseModel):
    """Input schema for the SimpleSearchTool."""
    query: str = Field(description="should be a search query")


class MultiplyNumbersArgs(BaseModel):
    """Input schema for the MultiplyNumbersTool."""
    x: float = Field(description="First number to multiply")
    y: float = Field(description="Second number to multiply")


# Custom tool for performing a simple web search using Tavily API.
class SimpleSearchTool(BaseTool):
    """Tool that uses Tavily Search to find information on the web."""
    # The name and description fields are overridden from the BaseTool class.
    # Pydantic v2 requires type annotations for overridden fields.
    name: str = "simple_search"
    description: str = "useful for when you need to answer questions about current events"
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """Use the tool."""
        # This method contains the core logic of the tool.
        try:
            from tavily import TavilyClient

            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                return "Tavily API key is not set in environment variables."

            client = TavilyClient(api_key=api_key)
            results = client.search(query=query)
            return f"Search results for: {query}\n\n\n{results}\n"
        except ImportError:
            return "Tavily client is not installed. Please install with `pip install tavily-python`."
        except Exception as e:
            return f"An error occurred during search: {e}"

# Custom tool for multiplying two numbers.
class MultiplyNumbersTool(BaseTool):
    """Tool that multiplies two numbers."""
    # The name and description fields are overridden from the BaseTool class.
    # Pydantic v2 requires type annotations for overridden fields.
    name: str = "multiply_numbers"
    description: str = "useful for multiplying two numbers"
    args_schema: Type[BaseModel] = MultiplyNumbersArgs

    def _run(
        self,
        x: float,
        y: float,
        **kwargs: Any,
    ) -> str:
        """Use the tool."""
        # This method performs the multiplication.
        result = x * y
        return f"The product of {x} and {y} is {result}"


# Create a list of tools that the agent can use.
tools = [
    SimpleSearchTool(),
    MultiplyNumbersTool(),
]

# Initialize a ChatOpenAI model. Using a powerful model like gpt-4o is recommended for agent tasks.
# Ensure your OPENAI_API_KEY is set in your environment.
llm = ChatOpenAI(model="gpt-4o")

# Pull a pre-built prompt template from LangChain Hub.
# This prompt is specifically designed for creating agents that use OpenAI's function-calling capabilities.
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create the agent using the create_tool_calling_agent function.
# This function combines the language model, the tools, and the prompt to create an agent.
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# Create the agent executor. The executor is responsible for running the agent and handling the interaction loop.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # Set to True to see the agent's thought process.
    handle_parsing_errors=True, # Helps in debugging by gracefully handling parsing errors.
)

# Test the agent with a sample query that requires the search tool.
print("--- Invoking Agent for Search Task ---")
response_search = agent_executor.invoke({"input": "Search for Apple Intelligence"})
print("\n--- Final Response ---")
print(response_search)


# Test the agent with a sample query that requires the multiplication tool.
print("\n--- Invoking Agent for Multiplication Task ---")
response_multiply = agent_executor.invoke({"input": "Multiply 10 and 20"})
print("\n--- Final Response ---")
print(response_multiply)
