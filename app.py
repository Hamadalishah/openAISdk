
import os
import asyncio

import chainlit as cl
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

# Load env vars
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

# Configure Gemini (via OpenAI-compatible endpoint)
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_tracing_disabled(disabled=True)

# Initialize your Agent once
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. ",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
)

@cl.on_message
async def main(message: cl.Message):
    user_text = message.content

    # Run your Agent on the incoming message
    result = await Runner.run(agent, user_text)

    # Send back the Agent's final output
    await cl.Message(
        content=result.final_output
    ).send()
