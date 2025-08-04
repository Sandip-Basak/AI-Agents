from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

import os
import random

model = LiteLlm(
    model='openai/gpt-3.5-turbo',
    api_key=os.getenv('OPENAI_API'),
)

def get_dad_joke():
    jokes = [
        "Why did the chicken cross the road? To get to the other side!",
        "What do you call a belt made of watches? A waist of time.",
        "What do you call fake spaghetti? An impasta!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
    ]
    return random.choice(jokes)

root_agent = Agent(
    model=model,
    name='dad_joke_agent',
    description='Dad joke agent',
    instruction="""
    You are a helpful assistant that can tell dad jokes. 
    Only use the tool `get_dad_joke` to tell jokes.
    """,
    tools=[get_dad_joke]
)
