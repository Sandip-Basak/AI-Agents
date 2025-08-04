# from dotenv import load_dotenv
# import os

# load_dotenv()
# os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# from google.genai import types
# from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
# from google.adk.agents.llm_agent import LlmAgent
# from google.adk.sessions import InMemorySessionService
# from google.adk.runners import Runner

# import asyncio

# async def get_agent():
#     tools, exit_stack = await MCPToolset.from_server(
#         connection_params=StdioServerParameters(
#             command="npx",
#             args=[
#                 "-y",
#                 "@openbnb/mcp-server-airbnb",
#                 "--ignore-robots-txt"
#             ]
#         )
#     )

#     agent = LlmAgent(
#         name="booking_agent",
#         tools=tools,
#         model="gemini-2.5-flash", 
#         instruction="You are a booking assistant for Airbnb. Help the user to find the listing of the places to stay."
#     )

#     return agent, exit_stack

# async def main():
#     agent, exit_stack = await get_agent()
#     session_service = InMemorySessionService()
#     session = session_service.create_session(
#         state={},
#         app_name="mcp_booking_app",
#         user_id="user_booking"
#     )
#     query = "What are the listings available for 2 people in Paris on august 1st to 4th 2025?"
#     content = types.Content(role="user", parts=[types.Part(text=query)])

#     runner = Runner(
#             app_name="mcp_booking_app",
#             agent=agent,
#             session_service=session_service
#         )
#     response=runner.run_async(
#         session_id=session.id,
#         user_id=session.user_id,
#         new_message=content
#     )

#     async for message in response:
#         print(message)

#     await exit_stack.aclose()

# if __name__ == "__main__":
#     asyncio.run(main())
    
    
import asyncio
# Import necessary classes for ADK agent, MCP tools, Runner, and Sessions
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types # Used for defining user messages (Content objects)
from dotenv import load_dotenv
import os
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Define the instruction for your agent
AIRBNB_INSTRUCTION = "You are an Airbnb enquiry agent, you can look for Airbnb listings using the tool."

async def create_airbnb_agent():
    """
    Asynchronously creates an LlmAgent with the Airbnb MCPToolset.
    This function sets up the connection to the @openbnb/mcp-server-airbnb server.
    """
    print("Creating Airbnb MCPToolset...")
    # MCPToolset manages the connection to the MCP server.
    # It requires an asynchronous environment for proper initialization and tool discovery [2, 3].
    airbnb_toolset = MCPToolset(
        connection_params=StdioServerParameters(
            command="npx", # Command to run the MCP server [4-8]
            args=[
                "-y", # Argument for npx to auto-confirm install [6]
                "@openbnb/mcp-server-airbnb", # The specific Airbnb MCP server package [5]
                "--ignore-robots-txt" # Configurable option to bypass robots.txt for testing [5]
            ]
        )
    )
    
    print("Initializing LlmAgent with the Airbnb MCPToolset...")
    # The LlmAgent expects instances of BaseTool in its 'tools' list.
    # MCPToolset, once properly initialized, functions as a proxy for the MCP server's tools [3].
    root_agent = LlmAgent(
        model="gemini-2.5-flash",
        name="Airbnb_Enquiry_Agent", # Renamed for consistency
        instruction=AIRBNB_INSTRUCTION,
        tools=[airbnb_toolset], # MCPToolset is directly provided to the agent's tools list [1, 6-8]
    )
    
    print(f"Agent '{root_agent.name}' successfully initialized.")
    return root_agent, airbnb_toolset # Return toolset for proper cleanup later

async def main_with_runner():
    """
    Main asynchronous function to create, run, and manage the agent using ADK's Runner.
    This demonstrates how an agent interacts with user queries over a session.
    """
    # 1. Initialize Session Service:
    # The InMemorySessionService is used to manage conversational state for the agent [1].
    # In a real application, you might use a persistent session service.
    session_service = InMemorySessionService()
    
    # 2. Create a Session:
    # A session represents a continuous conversation or interaction with the agent [1].
    session = session_service.create_session(
        state={}, # Initial state for the session
        app_name='airbnb_enquiry_app',
        user_id='user_airbnb_enquirer' # Unique identifier for the user
    )
    
    print(f"\nSession '{session.id}' created for user '{session.user_id}'.")
    
    # 3. Create the Agent and MCPToolset:
    # Call the async function to get the initialized LlmAgent and its associated MCPToolset.
    root_agent, toolset = await create_airbnb_agent()
    
    # 4. Initialize the Runner:
    # The Runner orchestrates the interaction between the user, the session, and the agent [1].
    runner = Runner(
        app_name='airbnb_enquiry_app',
        agent=root_agent,
        session_service=session_service,
        # artifact_service=InMemoryArtifactService(), # Optional: as shown in [1] for artifact storage
    )
    
    print("Runner initialized. Ready to process user queries.")
    
    # 5. Define and Send a User Query:
    # The Airbnb MCP server provides tools like 'airbnb_search' and 'airbnb_listing_details' [9, 10].
    # We will formulate a query that should trigger the 'airbnb_search' tool.
    user_query = "Find me Airbnb listings in London for 2 adults checking in on 2024-12-20 and checking out on 2024-12-25, with a max price of 300."
    print(f"\nUser Query: '{user_query}'")
    
    # Create a Content object representing the user's message [1]
    content = types.Content(role='user', parts=[types.Part(text=user_query)])
    
    print("Running agent with the query...")
    # 6. Run the Agent Asynchronously:
    # The run_async method sends the new message to the agent within the specified session [1].
    # It returns an async generator for events, allowing you to stream responses.
    events_async = runner.run_async(
        session_id=session.id,
        user_id=session.user_id,
        new_message=content
    )
    
    # 7. Process Events from the Agent:
    # Iterate through the events the agent generates during its execution [1].
    # This includes agent thoughts, tool calls, tool outputs, and final responses.
    async for event in events_async:
        print(f"Event received: {event}")
        # In a full application, you would parse 'event' objects to display
        # the agent's progress and final response to the user.
        # For example, if an event contains the agent's output message:
        # if hasattr(event, 'output') and event.output and event.output.role == 'model':
        #     print(f"Agent Response: {event.output.parts.text}")
    
    # 8. Clean Up:
    # It is crucial to properly close the MCPToolset connection when your application finishes [1, 11].
    # This terminates any subprocesses started by the toolset (like the npx server).
    print("\nClosing MCPToolset connection...")
    await toolset.close()
    print("Cleanup complete.")

if __name__ == "__main__":
    print("Launching asynchronous ADK Runner process for Airbnb agent...")
    try:
        # Start the asyncio event loop and run the main asynchronous function
        asyncio.run(main_with_runner())
    except Exception as e:
        print(f"An error occurred during agent execution: {e}")
    # Reminder: For 'npx' command to work, Node.js must be installed on your system [4, 12].   