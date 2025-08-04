from google.adk.agents import Agent
from google.adk.tools import google_search
from datetime import datetime
# from google.adk.tools import built_in_code_execution

def get_current_time() -> dict:
    """
    Get the current time in the format YYYY-MM-DD HH:MM:SS
    """
    return {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


root_agent = Agent(
    model='gemini-2.0-flash',
    name='toot_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
    # tools=[google_search]
    # tools=[built_in_code_execution]
    tools=[get_current_time]
)
