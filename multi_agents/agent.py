from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent

from shared import build_agent_generation_config, build_agent_model

AGENT_DIR = Path(__file__).resolve().parent

load_dotenv(dotenv_path=AGENT_DIR.parent / ".env", override=False)
load_dotenv(dotenv_path=AGENT_DIR / ".env", override=True)

AGENT_MODEL = build_agent_model("multi_agents", default_provider="azure")
AGENT_GENERATION_CONFIG = build_agent_generation_config("multi_agents")


def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
              
    """
    print(f"--- Tool: get_weather called for city: {city} ---")  # Log tool execution
    city_normalized = city.lower().replace(" ", "")  # Basic normalization

    # Mock weather data
    # api call
    mock_weather_db = {
        "newyork": {
            "status": "success",
            "report": "The weather in New York is sunny with a temperature of 25°C.",
        },
        "london": {
            "status": "success",
            "report": "It's cloudy in London with a temperature of 15°C.",
        },
        "tokyo": {
            "status": "success",
            "report": "Tokyo is experiencing light rain and a temperature of 18°C.",
        },
        "paris": {
            "status": "success",
            "report": "The weather in Paris is sunny with a temperature of 22°C.",
        },
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {
            "status": "error",
            "error_message": f"Sorry, I don't have weather information for '{city}'.",
        }

root_agent = Agent(
    name="travel_planner_agent",
    model=AGENT_MODEL,
    generate_content_config=AGENT_GENERATION_CONFIG,
    description="An agent that helps users plan their travel itineraries.",
    instruction="You are a travel planner assistant. You can help users plan their travel itineraries by providing information about",
    tools=[get_weather])
