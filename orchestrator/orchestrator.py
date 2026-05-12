from agentscope.agent import ReActAgent, UserAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.tool import Toolkit, execute_python_code, execute_shell_command
import os, asyncio

async def main():
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)
    toolkit.register_tool_function(execute_shell_command)

    weather_agent = ReActAgent(
        name="Weather Agent",
        sys_prompt="Your task is to be a helpful weather agent and get the weather by getting the temperature from the temperature service and then querying the database for the corresponding weather.",
        model=OpenAIChatModel(
            model_name="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            stream=True,
        ),
        formatter=OpenAIChatFormatter(),
        toolkit=toolkit,
    )

    user = UserAgent(name="user")

    msg = None
    while True:
        msg = await user(msg)
        if msg.get_text_content() == "done":
            break
        msg = await weather_agent(msg)

    

asyncio.run(main())
