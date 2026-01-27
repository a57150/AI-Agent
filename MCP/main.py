import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

SYSTEM_PROMPT = """
你是一個 tool-using agent。

你可以使用下列 tools：
- get_alerts(state: string)
- get_forecast(latitude: float, longitude: float)

【規則】
- 如果需要使用工具，請輸出 JSON
- 不要解釋
- 格式必須完全符合

{"tool": "tool_name", "arguments": { ... }}

如果不需要工具，直接用自然語言回答。
"""

async def run_agent(user_input: str):
    # 1️⃣ 問 LLM 要不要用工具
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
    )

    content = response.choices[0].message.content.strip()
    print("LLM output:\n", content)

    # 2️⃣ 嘗試解析成 tool call
    try:
        tool_call = json.loads(content)
    except json.JSONDecodeError:
        # 不用工具，直接回
        print("\nFinal answer:")
        print(content)
        return

    # 3️⃣ 啟動 MCP client
    server_params = StdioServerParameters(
        command="python",
        args=["weather.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                tool_call["tool"],
                arguments=tool_call["arguments"]
            )

            tool_result = result.content[0].text

    print("\nTool result:")
    print(tool_result)

    # 4️⃣ 把 tool 結果丟回 LLM 做最終回覆
    final_response = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324",
        messages=[
            {"role": "system", "content": "你是助理，請根據工具結果回答使用者"},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": content},
            {"role": "tool", "content": tool_result},
        ],
    )

    print("\nFinal answer:")
    print(final_response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(run_agent("華盛頓今天天氣怎麼樣？"))
