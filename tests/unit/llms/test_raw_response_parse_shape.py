"""`.parse()` shape differs across the openai SDK range this package supports.

`pyproject.toml` allows `openai = "^1.0.0"`. On 1.x, `.with_raw_response` hands
back a `LegacyAPIResponse` whose `.parse()` is synchronous. On 2.x it is an
`AsyncAPIResponse` whose `.parse()` is a coroutine. Awaiting unconditionally
works in a 2.x dev environment and raises

    object ChatCompletion can't be used in 'await' expression

on every 1.x deployment — which is exactly how it reached production. These
tests pin the shape-agnostic handling so a dev/prod SDK split cannot silently
reintroduce it.
"""

import inspect

import pytest


class _Message:
    content = "ok"


class _Choice:
    message = _Message()


class _Completion:
    choices = [_Choice()]
    usage = None
    id = "resp-1"


class LegacyRawResponse:
    """openai 1.x: .parse() returns the model directly."""

    headers = {"X-Upstream-Request-ID": "upstream-1"}

    def parse(self):
        return _Completion()


class AsyncRawResponse:
    """openai 2.x: .parse() is a coroutine."""

    headers = {"X-Upstream-Request-ID": "upstream-2"}

    async def parse(self):
        return _Completion()


async def _resolve(raw):
    """The handling under test, mirroring AIGatewayLLM._chat_simple."""
    parsed = raw.parse()
    return await parsed if inspect.isawaitable(parsed) else parsed


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [LegacyRawResponse(), AsyncRawResponse()])
async def test_parse_resolves_on_both_sdk_shapes(raw):
    response = await _resolve(raw)
    assert response.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_awaiting_unconditionally_is_the_bug_being_guarded():
    """The regression, stated as an executable fact rather than a comment."""
    with pytest.raises(TypeError, match="can't be used in 'await' expression"):
        await LegacyRawResponse().parse()
