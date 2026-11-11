# A local model API on Posit Connect

Two pieces of content on one Connect server:

```
Shiny app (shiny/)  --HTTP-->  model API (api/, edgemodelr::edge_api())  -->  GGUF model, CPU
```

The API loads the model once. Its engine generates up to `EDGEMODELR_SLOTS`
answers at the same time, and queues the rest first come first served, so
50 users at once get their answers later rather than errors. The Shiny app
never loads a model: it submits a job and polls for the text, so it stays
responsive however long the queue is.

## 1. Install edgemodelr from GitHub where you deploy from

Connect installs the packages your local library records. Install
edgemodelr with `remotes::install_github("PawanRamaMali/edgemodelr")`
before deploying, so the manifest points at GitHub.

The default build is right for Connect. It is portable, and it switches to
AVX2 kernels at run time on CPUs that have them. An `EDGEMODELR_SIMD=AVX2`
build was no faster in our tests, so a custom execution image is not needed
for speed.

## 2. Deploy the API

```r
rsconnect::deployAPI("api", appTitle = "Local model API")
```

In the content settings:

**Vars**

| Variable | Value |
| --- | --- |
| `EDGEMODELR_MODEL` | Path of a .gguf file on the server, on a persistent volume. Or leave it unset and set the next three. |
| `EDGEMODELR_MODEL_REPO`, `EDGEMODELR_MODEL_FILE` | Hugging Face repo and file to download on first start (default `unsloth/gemma-3-1b-it-GGUF`, `gemma-3-1b-it-Q4_K_M.gguf`). |
| `EDGEMODELR_CACHE_DIR` | A persistent path for the download. Each deploy starts in a fresh folder, so a model saved inside the bundle folder is downloaded again every time. |
| `EDGEMODELR_SLOTS` | Answers generated at once (default 8). |
| `EDGEMODELR_CTX` | Tokens per answer, prompt included (default 2048). |
| `EDGEMODELR_API_KEY` | Optional. Clients send it as `X-API-Key`. Connect uses the `Authorization` header for its own keys. |

**Runtime** (or `PATCH /v1/content/{guid}` with these fields)

| Setting | Value | Why |
| --- | --- | --- |
| Min processes | 1 | Keeps the model loaded. A cold start reloads it, and containers start slowly. |
| Max processes | 1 | Each process loads its own copy of the model. One process with many slots uses the CPU better than several processes. |
| Max connections per process | 200 | Default 20. With the default, Connect starts a second process (a second model) at about 11 users. |
| Load factor | 1.0 | Default 0.5, which makes Connect start processes early. |
| Initial timeout | 600 | Model download and load on first start. |
| Idle timeout | 3600 or more | Keeps the process, and the model, between bursts of use. |
| Read timeout | 3600 | A long queue can keep a plain chat request open for minutes. Jobs avoid this. |

If a proxy or load balancer sits in front of Connect, check its own read
timeout, often 60 seconds, for requests that stay open: `POST
/v1/chat/completions` waits for the whole answer. `POST /v1/jobs` returns at
once and is not affected.

**CPU and memory.** Give the API whole cores: the engine uses one thread per
core, and it follows the container's CPU limit. Memory is the model file
plus the KV cache, `EDGEMODELR_SLOTS * EDGEMODELR_CTX` tokens in q8_0. For
Gemma 3 1B (Q4_K_M) with 8 slots of 2048 tokens that is about 0.8 GB for the
model and 0.3 GB for the cache. Qwen3 1.7B needs about 1.1 GB plus 1 GB.

## 3. Deploy the Shiny app

```r
rsconnect::deployApp("shiny", appTitle = "Local model chat")
```

Set `EDGEMODELR_API_URL` to the API's content URL
(`https://connect.example.com/content/<guid>`), and `EDGEMODELR_API_KEY` if
the API has one. Connect gives the app its own `CONNECT_API_KEY`, which the
app sends so it can reach an API whose access is restricted to some users.
That key belongs to the app's owner, who must have access to the API.

The app may use Connect's defaults: it holds no model, and each session
only polls the API every half second while an answer is being written.

## Endpoints

| Method | Path | |
| --- | --- | --- |
| POST | `/v1/chat/completions` | OpenAI chat completions, with tools, `response_format` and `stream` |
| POST | `/v1/completions` | OpenAI text completions |
| POST | `/v1/jobs` | A chat completion run in the background: returns 202 and a job id at once |
| GET | `/v1/jobs/<id>` | Status, queue position, text so far, then the full response |
| DELETE | `/v1/jobs/<id>` | Cancel |
| GET | `/v1/models`, `/metrics` | Model name, engine counters |
| GET | `/health` | No key needed |

Any OpenAI client works against `<content URL>/v1`, for example
`ellmer::chat_openai_compatible()`.

## What to expect

On an 8-core laptop CPU (AMD Ryzen AI 7 PRO 350, with another program
using about half of it), Gemma 3 1B with 8 slots:

| | |
| --- | --- |
| One user alone | 5 to 6 s for a 64-token answer |
| 50 users at once, chat | all answered; median 47 s, slowest 83 s; about 28 tokens/s in total |
| 50 users at once, jobs | all answered; submit under 1 s; text visible while written; slowest 98 s |
| Throughput, 1 slot vs 8 slots | 13 vs 44 to 54 tokens/s in total |

Wait time grows with the number of users waiting, not the other way round:
a burst of users waits longer, and nobody gets an error until the queue
(512 requests here) is full. More cores shorten the wait almost linearly;
more slots beyond about 8 add little on CPU.
