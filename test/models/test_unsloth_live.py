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
"""Opt-in end-to-end check against a running Unsloth server.

Skipped unless UNSLOTH_BASE_URL and UNSLOTH_API_KEY are set. Sends one chat
completion with a tool definition, which is how OASIS agents choose actions.
"""
import os

import pytest

from oasis.models.unsloth import UnslothModelError, make_unsloth_model

pytestmark = pytest.mark.skipif(
    not (os.environ.get("UNSLOTH_BASE_URL")
         and os.environ.get("UNSLOTH_API_KEY")),
    reason="UNSLOTH_BASE_URL / UNSLOTH_API_KEY not set",
)

TOOL = {
    "type": "function",
    "function": {
        "name": "create_post",
        "description": "Publish a new post on the social platform.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Post text."}
            },
            "required": ["content"],
        },
    },
}


def test_unsloth_server_answers_with_tool_call():
    try:
        model = make_unsloth_model(model_config_dict={
            "temperature": 0,
            "tools": [TOOL],
        })
    except UnslothModelError as exc:
        pytest.fail(f"Unsloth server not usable: {exc}")

    response = model.run([{
        "role": "system",
        "content": "You are a social media user. Use tools to act."
    }, {
        "role": "user",
        "content": "Post exactly: Hello from OASIS."
    }])

    message = response.choices[0].message
    assert message.tool_calls, "expected the model to call create_post"
    assert message.tool_calls[0].function.name == "create_post"
