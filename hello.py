import asyncio
from openai import AsyncOpenAI 
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled  
from dotenv import load_dotenv 
import os
# Load environment variables from .env file
load_dotenv()
# Set your Gemini API key here


gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

set_tracing_disabled(disabled=True)

async def main():
    # This agent will use the custom LLM provider
    agent = Agent(
        name="Assistant",
        instructions="You only response when ask question to about America",
        model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    )

    result = await Runner.run(
        agent,
        "What is the capital of america?", 
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
    
