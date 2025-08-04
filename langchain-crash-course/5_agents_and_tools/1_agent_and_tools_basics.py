from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables from .env file
load_dotenv()


# Define a very simple tool function that returns the current time
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime  # Import datetime module to get current time

    now = datetime.datetime.now()  # Get current time
    return now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format


# List of tools available to the agent
tools = [
    Tool(
        name="Time",  # Name of the tool
        func=get_current_time,  # Function that the tool will execute
        # Description of the tool
        description="Useful for when you need to know the current time",
    ),
]

# Pull the prompt template from the hub
# ReAct = Reason and Action
# https://smith.langchain.com/hub/hwchase17/react
# prompt = hub.pull("hwchase17/react")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. You have access to the following tools:\n\n"
            "{tools}\n\n"
            "The user will ask you questions, and you should answer them using the tools available."
            " If you need to use a tool, use the following format:\n"
            "```json\n"
            "{{ \"action\": \"tool_name\", \"action_input\": \"input_for_tool\" }}\n"
            "```\n"
            "Only use the above JSON format for tool calls. Respond with only a single JSON object for a tool call."
            " If you don't need a tool, just respond directly to the user.\n"
            "Allowed tools: {tool_names}" # Important for the LLM to know which tools it can call
        ),
        MessagesPlaceholder("agent_scratchpad"), # This is where the agent's thoughts/actions/observations go
        ("user", "{input}"), # This is the user's input
    ]
)

# Initialize a ChatOpenAI model
llm = ChatOpenAI(
    model="gpt-4o", temperature=0
)

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

# Create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Run the agent with a test query
response = agent_executor.invoke({"input": "What time is it?"})

# Print the response from the agent
print("response:", response)
