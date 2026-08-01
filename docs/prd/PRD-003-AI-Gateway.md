# PRD-003: AI Gateway

## Status
Design-only PRD. No production code is produced in this document.

## Context

The current AI Gateway is an Axum HTTP server in `api-gateway/`. It exposes OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`), proxying to a single seed-node gRPC backend. It lacks authentication, authorization, streaming, batching, rate limiting, and usage accounting. This PRD designs a production-ready AI Gateway.

---

## 1. Problem

### 1.1 Current implementation

- `api-gateway/src/main.rs` mounts routes and enables CORS for all origins (`allow_origin(Any)`).
- `api-gateway/src/handlers/openai.rs` implements `chat_completions`, `embeddings`, `list_models`, and `get_model`.
- `chat_completions` returns `StatusCode::NOT_IMPLEMENTED` for `stream: true`.
- `api-gateway/src/handlers/predict.rs` provides a custom `POST /predict/{model_id}` endpoint.
- `api-gateway/src/payments/verifier.rs` verifies USDT payments but does not tie them to API keys or usage accounting.
- No authentication or rate limiting.

### 1.2 Attack vectors

- Open gateway: anyone can call `/v1/chat/completions` and consume orchestrator resources.
- No request limits: trivial DoS by sending many large sequences.
- CORS `Any` allows malicious websites to use a public gateway.
- No audit trail; cannot attribute usage or revenue to a customer.
- Single backend; gateway goes down if the one seed-node fails.

### 1.3 Architectural limitations

- Cannot route to multiple orchestrators.
- Cannot batch multiple client requests into a single backend inference call.
- Cannot stream tokens (relevant for future causal/text-generation models).
- No multi-region failover.

---

## 2. Target Architecture

### 2.1 Responsibilities

1. **OpenAI API compatibility** — `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`, `/v1/models/{id}`.
2. **Authentication** — API keys and optional JWT/OAuth.
3. **Authorization** — per-key scopes and per-model allow/deny lists.
4. **Rate limiting** — token bucket per key and per IP.
5. **Batching** — combine multiple client requests when safe.
6. **Streaming** — SSE streams for `stream: true`.
7. **Routing** — select orchestrator based on model, region, health, load.
8. **Usage accounting** — track tokens, requests, latency per key.
9. **Payments** — require on-chain payment or prepaid balance.
10. **Observability** — metrics, logs, tracing.

### 2.2 Trust boundaries

- Gateway does **not** hold model keys; it only forwards requests to orchestrators.
- Gateway authenticates clients but does not decrypt model artifacts.
- Gateway is stateless except for usage counters and optional billing state.

---

## 3. Alternatives

### 3.1 Option A: Extend current Axum gateway

**Pros:**
- Reuses existing code.
- Fast to ship.

**Cons:**
- Single Rust codebase becomes large.
- Harder to integrate with external identity providers.

### 3.2 Option B: Sidecar gateway with Envoy/Traefik + custom auth

Use Envoy for routing, rate limiting, and TLS termination; custom Rust service for payment validation and OpenAI request/response translation.

**Pros:**
- Production-grade routing and rate limiting.
- Easier multi-region.

**Cons:**
- More infrastructure to operate.
- Additional latency.

### 3.3 Recommended: Option C — Enhanced Axum gateway with pluggable middleware

Keep the Axum gateway but add layers for auth, rate limiting, batching, and routing. Add a separate `xenom-api-gateway` binary that can be run independently from orchestrator and consensus.

**Rationale:**
- Fits current stack.
- Easier to operate for node runners.
- Pluggable middleware allows replacing auth/rate-limit providers later.

---

## 4. API Changes

### 4.1 OpenAI compatibility

- `POST /v1/chat/completions` — support `stream: true` and `stream_options`.
- `POST /v1/embeddings` — support single and batch inputs.
- `GET /v1/models` and `GET /v1/models/{id}` — include `owned_by` and `permission`.

### 4.2 Authentication

Header: `Authorization: Bearer <api_key>`.

API key formats:
- `xkm_<base64>` — gateway-issued API key.
- `jwt_<token>` — JWT issued by external provider.
- `eth_<signature>` — signed Ethereum message.

### 4.3 New management endpoints

- `POST /admin/keys` — create API key.
- `DELETE /admin/keys/{id}` — revoke.
- `GET /admin/usage` — usage by key.
- `POST /admin/payments` — record on-chain payment.

### 4.4 gRPC to orchestrator

Keep current `seed_client.rs` but add:
- `block_height` param in `predict` and `get_model_info`.
- Connection pool to multiple orchestrators.
- Health-aware routing.

---

## 5. Blockchain Changes

- `InferencePayments.sol` remains but should emit an event indexed by a gateway-issued `apiKeyHash`.
- New optional contract: `ApiKeyRegistry` mapping `apiKeyHash` to `owner`, `rateLimit`, `credits`, `modelsAllowed`.
- `ModelGovernance.sol` may store `publicGateways` list for discoverability.

---

## 6. Storage Changes

- Gateway uses a small PostgreSQL/SQLite database for:
  - API key metadata.
  - Usage counters.
  - Prepaid balances.
- No model weights stored on the gateway.
- Usage logs can be archived to S3/Arweave for audit.

---

## 7. Security Changes

- API keys hashed in the database (Argon2 or HMAC-SHA256).
- Rate limiting: token bucket in Redis or in-memory.
- CORS restricted to configured origins.
- TLS termination with valid certificates.
- Input size limits (max sequence length, max tokens).
- Request timeout and circuit breaker for orchestrator calls.

---

## 8. Networking

- Gateway connects to one or more orchestrators via gRPC with mTLS.
- Health checks every 5s; unhealthy orchestrators removed from rotation.
- Multi-region: DNS geo-routing or anycast; gateway selects closest healthy orchestrator.
- Load balancing: least-loaded, round-robin, or latency-based.

---

## 9. Streaming

For `stream: true`:
- Open long-lived inference request to orchestrator.
- Stream tokens as Server-Sent Events (SSE) in OpenAI format.
- Each chunk is `data: {...}\n\n`.
- Final chunk contains `finish_reason`.

For backends that do not natively stream, the orchestrator simulates streaming by yielding the full response in one chunk.

---

## 10. Batching

- Buffer non-streaming requests for up to 50ms.
- Group requests by `(model_id, block_height)`.
- Merge inputs into a single backend call when the model supports batch inference.
- Return individual responses with original `query_id`.

---

## 11. Rate Limiting

- Global per-key token bucket: `requests_per_minute`, `tokens_per_minute`.
- Per-IP rate limit for unauthenticated endpoints.
- Model-level rate limit to protect rare/expensive models.
- Headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.

---

## 12. Migration Plan

1. Add auth middleware behind `XENO_GATEWAY_REQUIRE_AUTH` flag (default off).
2. Implement rate limiting in-memory first.
3. Add streaming support for `chat_completions`.
4. Add multi-orchestrator routing.
5. Add usage accounting and payment linkage.

---

## 13. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| DoS on open gateway | Critical | Require auth and rate limiting before public deployment. |
| Orchestrator overload | High | Batching, queue backpressure, circuit breakers. |
| API key leak | Medium | Short-lived keys, key rotation, hashed storage. |
| Multi-region inconsistency | Medium | Use same blockchain state; cache active model for 30s. |

---

## 14. Deliverables

This PRD is the design input for:
- `PRD-002-Orchestrator-Service.md`
- `PRD-008-Inference-Backends.md`
- `PRD-005-Encrypted-Storage.md`
