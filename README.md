# Free Model Aggregator

One Python 3.13 server (standard library only) that aggregates the fourteen
free OpenRouter models behind both completion wires, for the make.fun bounty
`free-model-aggregator-02`.

## What it does

- Serves `POST /v1/chat/completions` (OpenAI wire) and `POST /v1/messages`
  (Anthropic wire), in JSON and in streaming (SSE), plus `GET /v1/models` in
  the shape the request's auth header names (`Authorization: Bearer` gets the
  OpenAI list, `x-api-key` the Anthropic one).
- On start it reads the brief at `$MAKEFUN_PROMPT_FILE`, listens on a free
  port, and registers that port and its catalog with the env at
  `$MAKEFUN_ENV_URL/register`. `python3 main.py --list` prints the catalog
  document byte for byte as registered.
- Every prompt names `model: "auto"`. The server ranks the free models by the
  house's published rating, keeps those meeting the request's requirements
  (JSON mode, tools, images, context window), and dials the best through the
  gateway's Chat Completions route on the trial key. Anthropic requests are
  translated into Chat Completions and the answer translated back, with usage,
  stop reasons and tool calls whole.
- A refused call is retried once on the same model, then fails over down the
  ranking; an answer that comes back without the prompt's token is treated as
  a miss and the next model is tried. Recovered refusals are reported in the
  `x-makefun-upstream-failures` response header. The `model` field of every
  answer and chunk names the dialable id that really served the prompt.
- It polls `GET $MAKEFUN_ENV_URL/work` and exits 0 when the env says done.

Every choice is a function of the brief, the requests and the gateway's
answers: no wall clock and no randomness anywhere, so a replayed trial walks
exactly the same calls.

## Run

```
python3 main.py          # serve a trial (env vars set by the harness)
python3 main.py --list   # print the catalog document and exit
```
