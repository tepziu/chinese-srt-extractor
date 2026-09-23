# Batch folder translation TDD evidence

Date: 2026-09-19

## Scope

- Scan a local folder recursively and deterministically.
- Create a durable per-batch manifest with per-video checkpoints.
- Resume failed items without reprocessing completed items.
- Cancel and persist cancellation.
- Translate Chinese sidecar SRTs to English through the existing translation service.
- Add an opaque subtitle-band render mode that covers the complete selected region before burning English ASS subtitles.
- Reject changed source files, duplicate output stems, stale processing manifests, and hidden translation fallback/quality warnings in strict mode.
- Expose Web API and Web UI controls for starting, polling, stopping and resuming a batch.
- Support automatic per-video subtitle-region detection and a shared manual region selected by percentages or mouse drag on a preview frame; the same region is used for old-subtitle removal and English subtitle placement.

## Evidence

- RED: `tests/test_batch_pipeline.py` initially failed during collection because `services.batch_pipeline` did not exist.
- GREEN: `tests/test_batch_pipeline.py`, `tests/test_batch_api.py`, and `tests/test_batch_render.py` pass.
- Full Python suite: `150 passed`.
- Node UI regression suite: `3 passed`.
- Dependency check: `pip check` reports no broken requirements.
- Flask HTML smoke check confirms the batch tab, batch card, and start handler are present.

## Explicit limits

- The worker is sequential (`max_concurrency=1`) to protect the existing GPU/FFmpeg lease.
- Default strict mode refuses to render when the AI gateway is not configured or semantic translation falls back.
- A real provider-backed batch over user media was not run in this validation turn; use the configured gateway and inspect the manifest/artifacts after the first live batch.
