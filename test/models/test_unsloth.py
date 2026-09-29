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

import httpx
import pytest
from camel.models import OpenAICompatibleModel

from oasis.models.unsloth import (DEFAULT_UNSLOTH_URL, UnslothModelError,
                                  discover_unsloth_model, make_unsloth_model,
                                  normalize_unsloth_url)


def _client(models, status=200):
    """Return an httpx client whose GET /models answers with `models`."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = {"object": "list", "data": [{"id": m} for m in models]}
        return httpx.Response(status, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    client.calls = calls
    return client


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("UNSLOTH_BASE_URL", "UNSLOTH_API_KEY", "UNSLOTH_MODEL"):
        monkeypatch.delenv(var, raising=False)


class TestNormalizeUrl:

    def test_appends_v1(self):
        assert normalize_unsloth_url("http://localhost:8000") == \
            "http://localhost:8000/v1"

    def test_keeps_existing_v1_and_strips_slash(self):
        assert normalize_unsloth_url("http://host:1/v1/") == "http://host:1/v1"


class TestDiscoverModel:

    def test_returns_single_loaded_model(self):
        client = _client(["unsloth/gemma-3-4b-it-GGUF:Q4_K_M"])
        name = discover_unsloth_model("http://localhost:8000/v1",
                                      "sk-unsloth-x",
                                      client=client)
        assert name == "unsloth/gemma-3-4b-it-GGUF:Q4_K_M"
        req = client.calls[0]
        assert str(req.url) == "http://localhost:8000/v1/models"
        assert req.headers["authorization"] == "Bearer sk-unsloth-x"

    def test_no_models_loaded_is_an_error(self):
        with pytest.raises(UnslothModelError, match="no model is loaded"):
            discover_unsloth_model("http://localhost:8000/v1",
                                   "k",
                                   client=_client([]))

    def test_multiple_models_requires_explicit_choice(self):
        with pytest.raises(UnslothModelError, match="UNSLOTH_MODEL"):
            discover_unsloth_model("http://localhost:8000/v1",
                                   "k",
                                   client=_client(["a", "b"]))

    def test_unauthorized_is_reported(self):
        with pytest.raises(UnslothModelError, match="401"):
            discover_unsloth_model("http://localhost:8000/v1",
                                   "bad",
                                   client=_client(["a"], status=401))

    def test_connection_refused_is_reported(self):

        def handler(request):
            raise httpx.ConnectError("refused")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(UnslothModelError, match="unsloth run"):
            discover_unsloth_model("http://localhost:8000/v1",
                                   "k",
                                   client=client)


class TestMakeUnslothModel:

    def test_explicit_args(self):
        model = make_unsloth_model(model_type="my-model",
                                   url="http://gpu-box:9000",
                                   api_key="sk-unsloth-abc")
        assert isinstance(model, OpenAICompatibleModel)
        assert model.model_type == "my-model"
        assert str(model._client.base_url) == "http://gpu-box:9000/v1/"
        assert model._client.api_key == "sk-unsloth-abc"

    def test_env_defaults(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_BASE_URL", "http://127.0.0.1:8888/v1")
        monkeypatch.setenv("UNSLOTH_API_KEY", "sk-unsloth-env")
        monkeypatch.setenv("UNSLOTH_MODEL", "env-model")
        model = make_unsloth_model()
        assert model.model_type == "env-model"
        assert str(model._client.base_url) == "http://127.0.0.1:8888/v1/"
        assert model._client.api_key == "sk-unsloth-env"

    def test_default_url(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_API_KEY", "k")
        model = make_unsloth_model(model_type="m")
        assert str(model._client.base_url) == DEFAULT_UNSLOTH_URL + "/"

    def test_missing_api_key_is_an_error(self):
        with pytest.raises(UnslothModelError, match="UNSLOTH_API_KEY"):
            make_unsloth_model(model_type="m")

    def test_discovers_model_when_not_given(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_API_KEY", "k")
        model = make_unsloth_model(client_for_discovery=_client(["found"]))
        assert model.model_type == "found"

    def test_model_config_is_forwarded(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_API_KEY", "k")
        model = make_unsloth_model(model_type="m",
                                   model_config_dict={"temperature": 0.2})
        assert model.model_config_dict["temperature"] == 0.2
