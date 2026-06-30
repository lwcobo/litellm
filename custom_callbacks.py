import json
import os
from typing import Any, Literal, Optional, Tuple

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import DualCache, UserAPIKeyAuth

hook_target_host = os.environ.get("HOOK_TARGET_HOST", "http://localhost:8000")


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError, ValueError):
        return False


def _coerce_to_json_value(value: Any) -> Any:
    if _is_json_serializable(value):
        return value
    try:
        return str(value)
    except Exception:
        return f"<unserializable {type(value).__name__}>"


def _sanitize_for_json(value: Any, depth: int, max_depth: int) -> Tuple[bool, Any]:
    if depth > max_depth:
        return True, _coerce_to_json_value(value)

    if isinstance(value, dict):
        cleaned: dict = {}
        for k, v in value.items():
            if not _is_json_serializable(k):
                continue
            ok, cleaned_v = _sanitize_for_json(v, depth + 1, max_depth)
            if ok:
                cleaned[k] = cleaned_v
        return True, cleaned

    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            ok, cleaned_item = _sanitize_for_json(item, depth + 1, max_depth)
            if ok:
                cleaned_list.append(cleaned_item)
        return True, cleaned_list

    if isinstance(value, tuple):
        cleaned_tuple = []
        for item in value:
            ok, cleaned_item = _sanitize_for_json(item, depth + 1, max_depth)
            if ok:
                cleaned_tuple.append(cleaned_item)
        return True, cleaned_tuple

    return True, _coerce_to_json_value(value)


def _messages_contain_images(data: dict) -> bool:
    """Check if the request messages contain image content.

    Supports both message formats:
    - OpenAI:    {"type": "image_url", "image_url": {"url": "..."}}
    - Anthropic: {"type": "image",     "source": {"type": "base64", ...}}
    """
    _IMAGE_TYPES = {"image_url", "image"}

    messages = data.get("messages", [])
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in _IMAGE_TYPES:
                    return True
    return False


def _safe_merge(target: dict, source: dict) -> None:
    """Merge source into target, but never overwrite a value in target
    that is not JSON-serializable (i.e. an internal litellm object) with the
    stringified version coming back from the hook server."""
    for k, v in source.items():
        if k in target and not _is_json_serializable(target[k]):
            continue
        target[k] = v


async def send_hook(payload: dict) -> Optional[dict]:
    verbose_proxy_logger.info(f"send_hook with payload: {payload}")

    llm_data = payload.get("llm_data", {})
    if isinstance(llm_data, dict) and _messages_contain_images(llm_data):
        verbose_proxy_logger.info("send_hook: skipping, request contains image content")
        return None

    _, payload = _sanitize_for_json(payload, depth=0, max_depth=10)

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


class MyCustomHandler(CustomLogger):
    def __init__(self):
        super().__init__()

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
        result = await send_hook(
            {
                "method": "async_pre_call_hook",
                "llm_data": data,
                "call_type": call_type,
            }
        )
        if isinstance(result, dict):
            _safe_merge(data, result)
            return data
        return result

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

    async def async_moderation_hook(
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
