# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
r"""Connect OASIS agents to a model served by Unsloth.

Unsloth Studio, Unsloth Desktop and the ``unsloth run`` CLI expose the model
they have loaded as a local, OpenAI-compatible HTTP API::

    unsloth run --model unsloth/gemma-3-4b-it-GGUF:Q4_K_M

The server prints a base URL (``http://localhost:<port>``) and an API key
starting with ``sk-unsloth-``. Put them in ``UNSLOTH_BASE_URL`` and
``UNSLOTH_API_KEY`` and call :func:`make_unsloth_model`; the returned model
backend can be handed to ``generate_twitter_agent_graph`` or
``generate_reddit_agent_graph`` exactly like an OpenAI or vLLM model.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from camel.models import OpenAICompatibleModel

DEFAULT_UNSLOTH_URL = "http://localhost:8000/v1"

ENV_BASE_URL = "UNSLOTH_BASE_URL"
ENV_API_KEY = "UNSLOTH_API_KEY"
ENV_MODEL = "UNSLOTH_MODEL"

_START_HINT = ("Start the server with `unsloth run --model <model>` "
               "(or open Unsloth Studio / Desktop) and make sure "
               f"{ENV_BASE_URL} points at it.")


class UnslothModelError(RuntimeError):
    """Raised when the Unsloth server cannot be reached or configured."""


def normalize_unsloth_url(url: str) -> str:
    """Return ``url`` without a trailing slash and ending in ``/v1``."""
    url = url.strip().rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def discover_unsloth_model(url: str,
                           api_key: str,
                           *,
                           timeout: float = 10.0,
                           client: Optional[httpx.Client] = None) -> str:
    """Ask the Unsloth server which model it has loaded.

    Returns the model id when exactly one model is loaded. Raises
    :class:`UnslothModelError` otherwise, so the caller never silently talks
    to the wrong model.
    """
    url = normalize_unsloth_url(url)
    headers = {"Authorization": f"Bearer {api_key}"}
    own_client = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        try:
            response = client.get(f"{url}/models", headers=headers)
        except httpx.HTTPError as exc:
            raise UnslothModelError(
                f"Could not reach the Unsloth server at {url}: {exc}. "
                f"{_START_HINT}") from exc
    finally:
        if own_client:
            client.close()

    if response.status_code != 200:
        raise UnslothModelError(
            f"Unsloth server at {url} answered {response.status_code} "
            f"to GET /models. Check {ENV_API_KEY}: it must be the "
            "`sk-unsloth-...` key shown by Unsloth.")

    models = [m.get("id") for m in response.json().get("data", [])]
    models = [m for m in models if m]
    if not models:
        raise UnslothModelError(
            f"Unsloth server at {url} is running but no model is loaded. "
            f"{_START_HINT}")
    if len(models) > 1:
        raise UnslothModelError(
            f"Unsloth server at {url} has several models loaded "
            f"({', '.join(models)}). Choose one with {ENV_MODEL} or the "
            "`model_type` argument.")
    return models[0]


def make_unsloth_model(model_type: Optional[str] = None,
                       url: Optional[str] = None,
                       api_key: Optional[str] = None,
                       model_config_dict: Optional[Dict[str, Any]] = None,
                       *,
                       client_for_discovery: Optional[httpx.Client] = None,
                       **kwargs: Any) -> OpenAICompatibleModel:
    r"""Create a CAMEL model backend that talks to an Unsloth server.

    Args:
        model_type: Model id as reported by the server. Falls back to
            ``UNSLOTH_MODEL``; if neither is set the single loaded model is
            discovered automatically.
        url: Server base URL, e.g. ``http://localhost:8000``. Falls back to
            ``UNSLOTH_BASE_URL``, then ``http://localhost:8000/v1``.
        api_key: The ``sk-unsloth-...`` key. Falls back to
            ``UNSLOTH_API_KEY``. Required.
        model_config_dict: Generation parameters (temperature, max_tokens,
            tools, ...) forwarded to the chat completion request.
        client_for_discovery: Optional ``httpx.Client`` used only for the
            ``GET /models`` discovery call (mainly for tests).
        **kwargs: Forwarded to ``OpenAICompatibleModel`` (``timeout``,
            ``max_retries``, ...).

    Returns:
        An ``OpenAICompatibleModel`` ready to pass as ``model=`` to the OASIS
        agent-graph generators or to a ``ModelManager``.
    """
    url = normalize_unsloth_url(url or os.environ.get(ENV_BASE_URL)
                                or DEFAULT_UNSLOTH_URL)
    api_key = api_key or os.environ.get(ENV_API_KEY)
    if not api_key:
        raise UnslothModelError(
            f"No Unsloth API key. Set {ENV_API_KEY} to the `sk-unsloth-...` "
            "key printed by `unsloth run` (or created under Settings > API "
            "in Unsloth Studio), or pass api_key=.")

    model_type = model_type or os.environ.get(ENV_MODEL)
    if not model_type:
        model_type = discover_unsloth_model(url, api_key,
                                            client=client_for_discovery)

    return OpenAICompatibleModel(model_type=model_type,
                                 model_config_dict=model_config_dict,
                                 api_key=api_key,
                                 url=url,
                                 **kwargs)
