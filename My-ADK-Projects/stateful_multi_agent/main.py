import asyncio
from customer_service_agent.agent import customer_service_agent
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from datetime import datetime
from google.genai import types

load_dotenv()

session_service = InMemorySessionService()
initial_state = {
    "user_name": "Brandon Hancock",
    "purchased_courses": [],
    "interaction_history": [],
}

def update_interaction_history(session_service, app_name, user_id, session_id, entry):
    try:
        session = session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        interaction_history = session.state.get("interaction_history", [])
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        interaction_history.append(entry)
        updated_state = session.state.copy()
        updated_state["interaction_history"] = interaction_history

        session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state=updated_state,
        )
    except Exception as e:
        print(f"Error updating interaction history: {e}")

async def main_async():
    APP_NAME = "Customer Support"
    USER_ID = "aiwithbrandon"
    new_session = session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        state=initial_state,
    )
    SESSION_ID = new_session.id
    runner = Runner(
        agent=customer_service_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending conversation. Goodbye!")
            break
        update_interaction_history(session_service,APP_NAME,USER_ID,SESSION_ID,{"action":"user_query","query":user_input})
        try:
            content = types.Content(role="user", parts=[types.Part(text=user_input)])
            async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
                if event.author:
                    agent_name = event.author
                if event.is_final_response():
                    if (event.content and event.content.parts and hasattr(event.content.parts[0], "text") and event.content.parts[0].text):
                        final_response = event.content.parts[0].text.strip()
                        print(f"Agent: {final_response}")
                        update_interaction_history(session_service,APP_NAME,USER_ID,SESSION_ID,{"action": "agent_response","agent": agent_name,"response": final_response})
                        
        except Exception as e:
            print(f"Error during agent call: {e}")

if __name__ == "__main__":
    asyncio.run(main_async())