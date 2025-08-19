import json
import os
from typing import Literal, Optional

import httpx
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import DualCache, UserAPIKeyAuth

hook_target_host = os.environ.get("HOOK_TARGET_HOST", "http://localhost:8000")


async def send_hook(payload: dict) -> Optional[dict]:
    verbose_proxy_logger.info(f"send_hook with payload: {payload}")
    if 'litellm_logging_obj' in payload['llm_data']:
        del payload['llm_data']['litellm_logging_obj']
    
    # Create and use httpx client within the function
    async with httpx.AsyncClient(base_url=hook_target_host) as client:
        response = await client.post("/v1/gateway/litellm_hook/", json=payload)
        res_text = response.text
        verbose_proxy_logger.info(
            f"Response status {response.status_code}, response: {res_text}"
        )
        res_json = json.loads(res_text)
        if response.status_code in [200, 201]:
            return res_json["result"]
        else:
            return res_text


# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class MyCustomHandler(
    CustomLogger
):  # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    # Class variables or attributes
    def __init__(self):
        super().__init__()

    # LOG HOOKS

    def log_pre_api_call(self, model, messages, kwargs):
        pass

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    # CALL HOOKS - proxy only ####

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ):
        verbose_proxy_logger.info("async_pre_call_hook called")
        if call_type == "pass_through_endpoint":
            return
        return await send_hook(
            {
                "method": "async_pre_call_hook",
                "llm_data": data,
                "call_type": call_type,
            }
        )

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: Optional[str] = None,
    ):
        verbose_proxy_logger.info(
            f"async_post_call_failure_hook called, "
            f"request_data: {request_data}, original_exception: {original_exception}"
        )

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        verbose_proxy_logger.info(
            f"async_post_call_success_hook called, data={data}, response={response}"
        )

    async def async_moderation_hook(  # call made in parallel to llm api call
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ):
        verbose_proxy_logger.info("async_moderation_hook called")

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: str,
    ):
        verbose_proxy_logger.info(f"async_moderation_hook called, response: {response}")


proxy_handler_instance = MyCustomHandler()
