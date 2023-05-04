from enum import Enum
from typing import Any, Dict

import aiohttp
from loguru import logger
from oasst_backend.config import settings
from oasst_shared.exceptions import OasstError, OasstErrorCode


class OpenAIUrl(str, Enum):
    OPEN_AI_CHAT = "https://api.openai.com/v1/chat/completions"


class OpenAIChatModel(str, Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"

class OpenAIAPI:

    def __init__(self, api_url: str):
        self.api_url: str = api_url
        self.api_key: str = settings.OPENAI_API_KEY
        self.headers: Dict[str, str] = {"Authorization": f"Bearer {self.api_key}"}

    async def post(self, input: str, max_replies: str = 3) -> Any:
        async with aiohttp.ClientSession() as session:
            payload: Dict[str, str] = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": input}],
                "n": max_replies,
            }

            async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                # If we get a bad response
                if not response.ok:
                    logger.error(response)
                    logger.info(self.headers)
                    raise OasstError(
                        f"Response Error OpenAI API (Status: {response.status})",
                        error_code=OasstErrorCode.OPENAI_API_ERROR,
                    )

                # Get the response from the API call
                inference = await response.json()

        return inference
