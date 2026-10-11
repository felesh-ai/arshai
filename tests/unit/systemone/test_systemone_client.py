import json
import httpx
import pytest
from arshai.systemone import SystemOneClient, SystemOneConfig, DecisionUnavailable


def _client(handler, **kw):
    cfg = SystemOneConfig(base_url="https://gw.test/v1", api_key="k", model="jev-1.13.0", timeout=5.0, **kw)
    return SystemOneClient(cfg, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


@pytest.mark.asyncio
async def test_posts_model_state_questions_and_passes_answers_and_usage_through():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"], seen["auth"], seen["org"] = str(req.url), req.headers["authorization"], req.headers.get("x-auth-organization")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"answers": {"Q": {"probabilities": {"a": 0.8, "b": 0.2}}},
                                         "usage": {"input_tokens": 300, "output_tokens": 0, "credits": "0.01"}})

    res = await _client(handler, path="/decisions").decide(
        {"message": "m"}, {"Q": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}}},
        headers={"X-Auth-Organization": "acme"})
    assert seen["url"] == "https://gw.test/v1/decisions"
    assert seen["auth"] == "Bearer k" and seen["org"] == "acme"
    assert seen["body"]["model"] == "jev-1.13.0" and seen["body"]["state"] == {"message": "m"}
    assert res.answers["Q"]["probabilities"]["a"] == 0.8
    assert res.usage["credits"] == "0.01"


@pytest.mark.asyncio
async def test_default_path_is_systemone():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"answers": {}, "usage": {}})

    await _client(handler).decide({"message": "m"}, {})
    assert seen["url"] == "https://gw.test/v1/systemone"


@pytest.mark.asyncio
async def test_empty_criteria_dropped_without_mutating_caller():
    body = {}

    def handler(req):
        body.update(json.loads(req.content))
        return httpx.Response(200, json={"answers": {"N": {"noul": 0.4}}, "usage": {}})

    qs = {"N": {"type": "noul", "instructions": "i", "criteria": {}}}
    await _client(handler).decide({"message": "m"}, qs)
    assert "criteria" not in body["questions"]["N"]
    assert qs["N"]["criteria"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("resp", [httpx.Response(400, json={"error": "guardrail"}), httpx.Response(504),
                                  httpx.Response(200, content=b"not json"), httpx.Response(200, json={"usage": {}})])
async def test_any_failure_is_decision_unavailable(resp):
    with pytest.raises(DecisionUnavailable):
        await _client(lambda req: resp).decide({"message": "m"}, {"Q": {"type": "noul", "instructions": "i"}})


@pytest.mark.asyncio
async def test_transport_error_is_decision_unavailable_and_not_retried():
    calls = []

    def handler(req):
        calls.append(1)
        raise httpx.ConnectError("down")

    with pytest.raises(DecisionUnavailable):
        await _client(handler).decide({"message": "m"}, {"Q": {"type": "noul", "instructions": "i"}})
    assert len(calls) == 1
