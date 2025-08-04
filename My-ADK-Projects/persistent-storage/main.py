import asyncio
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from memory_agent.agent import memory_agent
from google.genai import types

load_dotenv()

db_url = "sqlite:///./my_agent_data.db"
session_service = DatabaseSessionService(db_url=db_url)
initial_state = {
    "user_name": "Sandip Basak",
    "reminders": [],
}

async def main_async():
    APP_NAME = "Memory Agent"
    USER_ID = "sandip_basak"

    existing_sessions = session_service.list_sessions(app_name=APP_NAME, user_id=USER_ID)

    if existing_sessions and len(existing_sessions.sessions)>0:
        SESSION_ID = existing_sessions.sessions[0].id
    else:
        new_session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, state=initial_state)
        SESSION_ID = new_session.id

    runner = Runner(agent=memory_agent,app_name=APP_NAME, session_service=session_service)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending conversation. Your data has been saved to the database.")
            break
        content=types.Content(role="user", parts=[types.Part(text=user_input)])
        final_response_text = None

        try:
            async for event in runner.run_async(user_id=USER_ID,session_id=SESSION_ID,new_message=content):
                if event.is_final_response():
                    if event.content and event.content.parts and hasattr(event.content.parts[0], "text") and event.content.parts[0].text:
                        final_response_text = event.content.parts[0].text.strip()
                        print(f"Agent: {final_response_text}")
        except Exception as e:
            print(f"Error during agent call: {e}")

if __name__ == "__main__":
    asyncio.run(main_async())