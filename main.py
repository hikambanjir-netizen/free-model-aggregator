#!/usr/bin/env python3
"""free-model-aggregator-02: one server, two client wires, the best free model per prompt.

A client points at this server and every completion it answers is a real call to a
free OpenRouter model, dialled through the grader's gateway on the trial's key.
For each prompt the server ranks the brief's fourteen free models by the house's
rating, keeps the ones that meet the request's requirements (JSON mode, tools,
images, context), dials the best, and fails over down the ranking when a call is
refused or answers without the prompt's token. Anthropic-wire requests are
translated into Chat Completions on the way out and the answer is translated
back, streamed or not as the client asked. The model field of every answer and
every streamed chunk names the dialable id that was really dialled.

Determinism: every choice is a function of the brief, the client's request and
the gateway's answers seen so far. No wall clock reaches a request or an answer
(fixed `created` values, counter-based ids), and there is no randomness at all.

Run as `python3 main.py` inside a trial, or `python3 main.py --list` to print
the catalog document exactly as it is registered.
"""

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------
# Environment and constants
# --------------------------------------------------------------------------

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
GATEWAY_URL = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", "")

PROTOCOL_VERSION = 1
MODELS_SCHEMA = "free-model-aggregator-02/models/v1"
PROVIDER_ID = "openrouter"
PROVIDER_BASE_URL = "https://openrouter.ai/api/v1"
PROVIDER_KEY_ENV = "OPENROUTER_API_KEY"
PROVIDER_WIRE = "openai"

MAX_OUTPUT_TOKENS = 8192          # the gateway refuses a larger cap
DEFAULT_MAX_TOKENS = 1024         # cap set when a client body names none
UPSTREAM_TIMEOUT_SEC = 20         # the client waits 30 s for an answer
POLL_INTERVAL_SEC = 0.25
MAX_ATTEMPTS = 4                  # gateway calls spent on one prompt at most
MAX_TRAIL_ROWS = 8                # the failures header holds at most 8 rows

# Fixed stamps: nothing in an answer may read the clock, so listings and
# synthesized chunks carry these constants instead of a time.
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"

# The shape of the unique token each prompt carries (the scorer's own shape).
TOKEN_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{3,12}-[0-9]{4})(?![0-9A-Z])")


def is_record(value):
    return isinstance(value, dict)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def dumps(value):
    """One deterministic JSON spelling: insertion order, ASCII, no spaces."""
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


# --------------------------------------------------------------------------
# The brief: the free model list, found inside the prompt file's prose
# --------------------------------------------------------------------------

def _object_end(text, start):
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _json_documents(text):
    """Every top-level JSON object embedded in `text`, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def load_models():
    """The brief's free models, as a list of plain dicts; [] without a brief."""
    if not PROMPT_FILE:
        return []
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return []
    for doc in _json_documents(text):
        if doc.get("schema") == MODELS_SCHEMA and isinstance(doc.get("models"), list):
            models = []
            for raw in doc["models"]:
                if not is_record(raw) or not isinstance(raw.get("id"), str):
                    continue
                models.append({
                    "id": raw["id"],
                    "display_name": raw.get("display_name") or raw["id"],
                    "rating_milli": raw.get("rating_milli") if is_number(raw.get("rating_milli")) else 0,
                    "context_length": raw.get("context_length") if is_number(raw.get("context_length")) else None,
                    "tools": raw.get("tools") is True,
                    "vision": raw.get("vision") is True,
                    "json": raw.get("json") is True,
                    "free": raw.get("free") is not False,
                })
            return models
    return []


def build_catalog(models):
    """The catalog document: one provider, exactly the free dialable ids."""
    rows = []
    for model in models:
        if not model["free"]:
            continue
        rows.append({
            "id": model["id"],
            "free": True,
            "context_length": model["context_length"],
        })
    return {"providers": [{
        "id": PROVIDER_ID,
        "base_url": PROVIDER_BASE_URL,
        "key_env": PROVIDER_KEY_ENV,
        "wire": PROVIDER_WIRE,
        "models": rows,
    }]}


# --------------------------------------------------------------------------
# What a request needs, and which free models may serve it
# --------------------------------------------------------------------------

def _text_of_content(content):
    """The plain text of a message content, whichever wire shaped it."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif is_record(block):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block.get("type") == "tool_result":
                    parts.append(_text_of_content(block.get("content")))
        return "\n".join(parts)
    return ""


def _content_has_image(content):
    if not isinstance(content, list):
        return False
    for block in content:
        if is_record(block) and block.get("type") in ("image", "image_url", "input_image"):
            return True
    return False


def request_needs(body, raw_len):
    """Requirements read off the client's own request, on either wire."""
    needs = {"tools": False, "vision": False, "json": False, "context": 0}
    if isinstance(body.get("tools"), list) and body["tools"]:
        needs["tools"] = True
    if is_record(body.get("response_format")):
        needs["json"] = True
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if is_record(message) and _content_has_image(message.get("content")):
                needs["vision"] = True
    max_tokens = body.get("max_tokens")
    if not is_number(max_tokens):
        max_tokens = body.get("max_completion_tokens")
    if not is_number(max_tokens):
        max_tokens = DEFAULT_MAX_TOKENS
    # A rough input size: a token per three bytes of request, plus the cap.
    needs["context"] = raw_len // 3 + int(max_tokens) + 64
    return needs


def request_token(body):
    """The prompt's unique token, read from the client's messages.

    The last user message that carries one names it; system text and earlier
    turns are only a fallback. None when the request carries no token.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    fallback = None
    chosen = None
    for message in messages:
        if not is_record(message):
            continue
        for match in TOKEN_RE.finditer(_text_of_content(message.get("content"))):
            fallback = match.group(1)
            if message.get("role") == "user":
                chosen = match.group(1)
    return chosen or fallback


def rank_candidates(models, needs):
    """Free models meeting the needs, best rating first, id as tie-break."""
    ranked = []
    for model in models:
        if not model["free"]:
            continue
        if needs["tools"] and not model["tools"]:
            continue
        if needs["vision"] and not model["vision"]:
            continue
        if needs["json"] and not model["json"]:
            continue
        context = model["context_length"]
        if context is not None and context < needs["context"]:
            continue
        ranked.append(model)
    ranked.sort(key=lambda m: (-m["rating_milli"], m["id"]))
    return ranked


def attempt_plan(ranked):
    """The dialling order for one prompt: best, best again, then down the list.

    A refusal is retried once on the same model (a rate limit can clear), then
    the next qualifying models in rating order, at most MAX_ATTEMPTS calls.
    The plan is a function of the ranking alone; how far it runs depends only
    on the gateway's answers, so a replay walks the same calls.
    """
    if not ranked:
        return []
    plan = [ranked[0]["id"], ranked[0]["id"]]
    for model in ranked[1:]:
        plan.append(model["id"])
    return plan[:MAX_ATTEMPTS]


# --------------------------------------------------------------------------
# Wire translation: Anthropic Messages <-> OpenAI Chat Completions
# --------------------------------------------------------------------------

def _clamp_max_tokens(value):
    if not is_number(value) or value <= 0:
        return DEFAULT_MAX_TOKENS
    return max(1, min(MAX_OUTPUT_TOKENS, int(value)))


def _anthropic_system_text(system):
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n".join(
            block["text"] for block in system
            if is_record(block) and isinstance(block.get("text"), str)
        )
    return ""


def _anthropic_image_part(block):
    source = block.get("source")
    if is_record(source):
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") or "image/png"
            return {"type": "image_url",
                    "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _anthropic_message_to_openai(message):
    """One Anthropic message as one or more Chat Completions messages."""
    role = message.get("role")
    if role not in ("user", "assistant"):
        role = "user"
    content = message.get("content")
    if isinstance(content, str) or content is None:
        return [{"role": role, "content": content or ""}]
    if not isinstance(content, list):
        return [{"role": role, "content": _text_of_content(content)}]
    out = []
    parts = []        # text/image parts of the message being built
    tool_calls = []   # assistant tool_use blocks of the message being built

    def flush():
        if parts or tool_calls or not out:
            msg = {"role": role}
            if parts and all(p.get("type") == "text" for p in parts):
                msg["content"] = "".join(p["text"] for p in parts)
            elif parts:
                msg["content"] = list(parts)
            else:
                msg["content"] = None
            if tool_calls:
                msg["tool_calls"] = list(tool_calls)
            out.append(msg)
        parts.clear()
        tool_calls.clear()

    for block in content:
        if not is_record(block):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})
        elif kind == "image":
            part = _anthropic_image_part(block)
            if part is not None:
                parts.append(part)
        elif kind == "tool_use":
            args = block.get("input") if is_record(block.get("input")) else {}
            tool_calls.append({
                "id": block.get("id") or "toolu_agg2",
                "type": "function",
                "function": {"name": block.get("name") or "",
                             "arguments": dumps(args)},
            })
        elif kind == "tool_result":
            flush()
            out.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id") or "",
                "content": _text_of_content(block.get("content")),
            })
    flush()
    return out


def _anthropic_tools_to_openai(tools):
    out = []
    for tool in tools:
        if not is_record(tool):
            continue
        function = {"name": tool.get("name") or "",
                    "parameters": tool.get("input_schema") or {"type": "object"}}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        out.append({"type": "function", "function": function})
    return out


def _anthropic_tool_choice_to_openai(choice):
    if not is_record(choice):
        return None
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and isinstance(choice.get("name"), str):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def anthropic_body_to_openai(body):
    """An Anthropic Messages body as a Chat Completions body (model unset)."""
    out = {"messages": []}
    system = _anthropic_system_text(body.get("system"))
    if system:
        out["messages"].append({"role": "system", "content": system})
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if is_record(message):
                out["messages"].extend(_anthropic_message_to_openai(message))
    out["max_tokens"] = _clamp_max_tokens(body.get("max_tokens"))
    if is_number(body.get("temperature")):
        out["temperature"] = body["temperature"]
    if is_number(body.get("top_p")):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list) and body["tools"]:
        out["tools"] = _anthropic_tools_to_openai(body["tools"])
        choice = _anthropic_tool_choice_to_openai(body.get("tool_choice"))
        if choice is not None:
            out["tool_choice"] = choice
    return out


def openai_body_for_upstream(body, model_id):
    """The client's own Chat Completions body, retargeted at `model_id`."""
    out = {"model": model_id}
    messages = body.get("messages")
    out["messages"] = messages if isinstance(messages, list) else []
    cap = body.get("max_tokens")
    if not is_number(cap):
        cap = body.get("max_completion_tokens")
    out["max_tokens"] = _clamp_max_tokens(cap)
    for key in ("temperature", "top_p", "top_k", "stop", "presence_penalty",
                "frequency_penalty", "response_format", "tools", "tool_choice",
                "seed", "logit_bias"):
        if key in body and body[key] is not None:
            out[key] = body[key]
    return out


# --------------------------------------------------------------------------
# The gateway's answer: usage, text, and translation back to the client wire
# --------------------------------------------------------------------------

def _usage_of(answer, request_body, text):
    """Whole usage counts for an answer, estimated only when it carries none.

    The estimate is a function of the request and the answer text alone, so a
    replay that hears the same answer fills the same numbers.
    """
    usage = answer.get("usage") if is_record(answer) else None
    if is_record(usage):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if is_number(prompt) and is_number(completion):
            return {"prompt_tokens": int(prompt), "completion_tokens": int(completion)}
    prompt = max(0, len(dumps(request_body)) // 3)
    completion = max(1, len(text) // 4)
    return {"prompt_tokens": prompt, "completion_tokens": completion}


def openai_answer_text(answer):
    choices = answer.get("choices") if is_record(answer) else None
    if isinstance(choices, list) and choices and is_record(choices[0]):
        message = choices[0].get("message")
        if is_record(message):
            return _text_of_content(message.get("content"))
    return ""


def _finish_reason_of(answer):
    choices = answer.get("choices") if is_record(answer) else None
    if isinstance(choices, list) and choices and is_record(choices[0]):
        reason = choices[0].get("finish_reason")
        if isinstance(reason, str) and reason:
            return reason
    return "stop"


OPENAI_TO_ANTHROPIC_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def openai_answer_to_anthropic(answer, model_id, msg_id, request_body):
    """A Chat Completions answer as an Anthropic message object."""
    text = openai_answer_text(answer)
    usage = _usage_of(answer, request_body, text)
    content = []
    if text:
        content.append({"type": "text", "text": text})
    choices = answer.get("choices") if is_record(answer) else None
    message = {}
    if isinstance(choices, list) and choices and is_record(choices[0]):
        if is_record(choices[0].get("message")):
            message = choices[0]["message"]
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        index = 0
        for call in calls:
            if not is_record(call) or not is_record(call.get("function")):
                continue
            index += 1
            function = call["function"]
            raw_args = function.get("arguments")
            parsed = None
            if isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                except ValueError:
                    parsed = None
            elif is_record(raw_args):
                parsed = raw_args
            if not is_record(parsed):
                parsed = {"value": parsed} if parsed is not None else {}
            content.append({
                "type": "tool_use",
                "id": call.get("id") or "toolu_agg2_%d" % index,
                "name": function.get("name") or "",
                "input": parsed,
            })
    if not content:
        content.append({"type": "text", "text": ""})
    reason = OPENAI_TO_ANTHROPIC_STOP.get(_finish_reason_of(answer), "end_turn")
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content,
        "stop_reason": reason,
        "stop_sequence": None,
        "usage": {"input_tokens": usage["prompt_tokens"],
                  "output_tokens": usage["completion_tokens"]},
    }


def openai_answer_for_client(answer, model_id, request_body):
    """The gateway's own answer, with the dialled id put back on it."""
    out = dict(answer) if is_record(answer) else {}
    out["model"] = model_id
    if not isinstance(out.get("choices"), list) or not out["choices"]:
        out["choices"] = [{
            "index": 0,
            "message": {"role": "assistant", "content": openai_answer_text(answer)},
            "finish_reason": "stop",
        }]
    text = openai_answer_text(out)
    out["usage"] = _usage_of(answer, request_body, text)
    return out


# --------------------------------------------------------------------------
# Streaming: a complete answer re-expressed as the wire's event stream
# --------------------------------------------------------------------------

def openai_stream_chunks(answer, model_id, answer_id, request_body):
    """Chat Completions SSE frames for a whole answer, in the API's order."""
    client = openai_answer_for_client(answer, model_id, request_body)
    choice = client["choices"][0]
    message = choice.get("message") if is_record(choice.get("message")) else {}
    frames = []

    def frame(payload):
        frames.append("data: " + dumps(payload) + "\n\n")

    base = {"id": answer_id, "object": "chat.completion.chunk",
            "created": FIXED_CREATED, "model": model_id}
    first = dict(base)
    first["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    frame(first)
    text = message.get("content") if isinstance(message.get("content"), str) else ""
    if text:
        chunk = dict(base)
        chunk["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
        frame(chunk)
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        for index, call in enumerate(calls):
            if not is_record(call):
                continue
            delta_call = {"index": index,
                          "id": call.get("id") or "call_agg2_%d" % index,
                          "type": "function",
                          "function": call.get("function") or {}}
            chunk = dict(base)
            chunk["choices"] = [{"index": 0, "delta": {"tool_calls": [delta_call]},
                                 "finish_reason": None}]
            frame(chunk)
    finish = dict(base)
    finish["choices"] = [{"index": 0, "delta": {},
                          "finish_reason": choice.get("finish_reason") or "stop"}]
    frame(finish)
    usage_chunk = dict(base)
    usage_chunk["choices"] = []
    usage_chunk["usage"] = client["usage"]
    frame(usage_chunk)
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_stream_events(message):
    """Messages SSE frames for a whole Anthropic message, in the API's order."""
    frames = []

    def frame(event, payload):
        frames.append("event: " + event + "\ndata: " + dumps(payload) + "\n\n")

    start = {
        "type": "message_start",
        "message": {
            "id": message["id"],
            "type": "message",
            "role": "assistant",
            "model": message["model"],
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": message["usage"]["input_tokens"],
                      "output_tokens": 0},
        },
    }
    frame("message_start", start)
    for index, block in enumerate(message["content"]):
        if block.get("type") == "text":
            frame("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "text", "text": ""}})
            if block.get("text"):
                frame("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "text_delta", "text": block["text"]}})
        elif block.get("type") == "tool_use":
            frame("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "tool_use", "id": block["id"],
                                  "name": block["name"], "input": {}}})
            frame("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta",
                          "partial_json": dumps(block["input"])}})
        frame("content_block_stop", {"type": "content_block_stop", "index": index})
    frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
        "usage": {"output_tokens": message["usage"]["output_tokens"]},
    })
    frame("message_stop", {"type": "message_stop"})
    return "".join(frames)


# --------------------------------------------------------------------------
# Dialling the gateway, with refusal failover
# --------------------------------------------------------------------------

def _error_code(raw, status):
    """The provider's own error code from a refusal body, or ''."""
    try:
        value = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return ""
    if is_record(value) and is_record(value.get("error")):
        error = value["error"]
        for key in ("code", "type"):
            code = error.get(key)
            if isinstance(code, str) and code:
                return code[:64]
        if is_number(error.get("code")):
            return str(int(error["code"]))
    return ""


def dial_gateway(upstream_body):
    """One Chat Completions call through the gateway.

    Returns (status, answer dict or None, raw body bytes). A transport error
    is reported as status 0 so the caller can fail over from it.
    """
    url = GATEWAY_URL + "/chat/completions"
    data = dumps(upstream_body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Authorization", "Bearer " + GATEWAY_KEY)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT_SEC) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as err:
        raw = err.read()
        status = err.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, None, b""
    answer = None
    try:
        parsed = json.loads(raw.decode("utf-8", "replace"))
        if is_record(parsed):
            answer = parsed
    except ValueError:
        answer = None
    return status, answer, raw


class Aggregator:
    """Serves the two client wires from the brief's free models."""

    def __init__(self, models):
        self.models = models
        self.counter = 0
        self.lock = threading.Lock()

    def next_id(self, prefix):
        with self.lock:
            self.counter += 1
            return "%s%d" % (prefix, self.counter)

    def serve(self, wire, body, raw_len):
        """Answer one completion request.

        Returns (status, content_type, payload text, trail rows). The trail
        holds every refusal this prompt recovered from, in order.
        """
        needs = request_needs(body, raw_len)
        token = request_token(body)
        ranked = rank_candidates(self.models, needs)
        if wire == "anthropic":
            base_body = anthropic_body_to_openai(body)
        else:
            base_body = openai_body_for_upstream(body, "")
            base_body.pop("model", None)
        trail = []
        fallback = None       # (model_id, answer) of a tokenless 2xx answer
        last_status = 502
        last_raw = b""
        for model_id in attempt_plan(ranked):
            # One key order for every call: model first, the rest as built.
            upstream = {"model": model_id}
            upstream.update(base_body)
            status, answer, raw = dial_gateway(upstream)
            if 200 <= status < 300 and answer is not None:
                text = openai_answer_text(answer)
                if token is None or token in text:
                    return self._answer(wire, body, model_id, answer, upstream, trail)
                if fallback is None:
                    fallback = (model_id, answer, upstream)
                continue
            last_status = status if status else 502
            last_raw = raw
            trail.append({"status": status if status else 502,
                          "code": _error_code(raw, status),
                          "model": model_id})
            if len(trail) >= MAX_TRAIL_ROWS:
                break
        if fallback is not None:
            model_id, answer, upstream = fallback
            return self._answer(wire, body, model_id, answer, upstream, trail)
        return self._give_up(wire, last_status, last_raw, trail)

    def _answer(self, wire, body, model_id, answer, upstream, trail):
        stream = body.get("stream") is True
        if wire == "anthropic":
            msg_id = self.next_id("msg_agg2_")
            message = openai_answer_to_anthropic(answer, model_id, msg_id, upstream)
            if stream:
                return 200, "text/event-stream", anthropic_stream_events(message), trail
            return 200, "application/json", dumps(message), trail
        answer_id = self.next_id("chatcmpl-agg2-")
        if stream:
            payload = openai_stream_chunks(answer, model_id, answer_id, upstream)
            return 200, "text/event-stream", payload, trail
        client = openai_answer_for_client(answer, model_id, upstream)
        client.setdefault("id", answer_id)
        client.setdefault("object", "chat.completion")
        client.setdefault("created", FIXED_CREATED)
        return 200, "application/json", dumps(client), trail

    def _give_up(self, wire, status, raw, trail):
        """A refusal in the client's own wire shape, with the trail attached."""
        code = _error_code(raw, status)
        if wire == "anthropic":
            payload = {"type": "error",
                       "error": {"type": "api_error",
                                 "message": "every free model refused this prompt"}}
            return status, "application/json", dumps(payload), trail
        payload = {"error": {"message": "every free model refused this prompt",
                             "type": "upstream_error",
                             "code": code or None}}
        return status, "application/json", dumps(payload), trail


# --------------------------------------------------------------------------
# The HTTP server: the two completion wires and the model list
# --------------------------------------------------------------------------

def models_payload(wire, models):
    """GET /v1/models in the shape the request's auth header names."""
    free = [m for m in models if m["free"]]
    if wire == "anthropic":
        data = [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                 "created_at": FIXED_CREATED_AT} for m in free]
        return {"data": data, "has_more": False,
                "first_id": data[0]["id"] if data else None,
                "last_id": data[-1]["id"] if data else None}
    data = [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
             "owned_by": PROVIDER_ID} for m in free]
    return {"object": "list", "data": data}


class Handler(BaseHTTPRequestHandler):
    aggregator = None     # set by main()
    catalog_models = []   # set by main()
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, payload, extra_headers=None):
        data = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status, value):
        self._send(status, "application/json", dumps(value))

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/v1/models", "/models"):
            if self.headers.get("x-api-key") is not None:
                wire = "anthropic"
            else:
                wire = "openai"
            self._send_json(200, models_payload(wire, self.catalog_models))
            return
        self._send_json(404, {"error": {"message": "no such route",
                                        "type": "invalid_request_error"}})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            wire = "openai"
        elif path == "/v1/messages":
            wire = "anthropic"
        else:
            self._send_json(404, {"error": {"message": "no such route",
                                            "type": "invalid_request_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            body = None
        if not is_record(body):
            if wire == "anthropic":
                self._send_json(400, {"type": "error", "error": {
                    "type": "invalid_request_error",
                    "message": "the request body is not one JSON object"}})
            else:
                self._send_json(400, {"error": {
                    "message": "the request body is not one JSON object",
                    "type": "invalid_request_error"}})
            return
        status, content_type, payload, trail = self.aggregator.serve(wire, body, len(raw))
        headers = {}
        if trail:
            headers["x-makefun-upstream-failures"] = dumps(trail[:MAX_TRAIL_ROWS])
        self._send(status, content_type, payload, headers)


# --------------------------------------------------------------------------
# The trial lifecycle: register, serve, poll /work, exit 0 when done
# --------------------------------------------------------------------------

def _env_request(method, path, payload=None, timeout=10):
    url = ENV_URL + path
    data = dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def register(port, catalog_text):
    """POST /register with the catalog embedded byte for byte as --list prints."""
    body = '{"protocol_version":%d,"port":%d,"catalog":%s}' % (
        PROTOCOL_VERSION, port, catalog_text)
    request = urllib.request.Request(ENV_URL + "/register",
                                     data=body.encode("utf-8"), method="POST")
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv):
    models = load_models()
    catalog_text = dumps(build_catalog(models))
    if "--list" in argv:
        sys.stdout.write(catalog_text + "\n")
        return 0
    Handler.aggregator = Aggregator(models)
    Handler.catalog_models = models
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if ENV_URL:
        for _ in range(40):
            try:
                register(port, catalog_text)
                break
            except Exception:
                time.sleep(0.5)
        while True:
            try:
                work = _env_request("GET", "/work", timeout=10)
            except Exception:
                work = None
            if is_record(work) and work.get("done") is True:
                break
            time.sleep(POLL_INTERVAL_SEC)
    else:
        try:
            thread.join()
        except KeyboardInterrupt:
            pass
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
