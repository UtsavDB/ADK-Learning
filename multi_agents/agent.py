from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

AGENT_MODEL = "ollama/gemma3:1b"

root_agent = Agent(
    name="travel_planner_agent",
    model=LiteLlm(model=AGENT_MODEL),
    description="An agent that helps users plan their travel itineraries.",
    instruction="You are a travel planner assistant. You can help users plan their travel itineraries by providing information about")