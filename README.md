# UnitAI

**Local AI for embedded and IoT systems.**

UnitAI orchestrates several small, domain-specific language models — one for hardware, one for firmware, one for debugging, one for vision — so the combined system responds like a full embedded systems engineer instead of a general-purpose chatbot.

🔗 **[Live site](https://unitaitr.github.io/UnitAi/)**

---

## How it works

A single general-purpose model has to be a little bit of everything, which makes it imprecise about any one part. UnitAI takes the opposite approach: specialized models, each scoped to one part of the job, coordinated by an orchestrator that routes each question to the right model and merges the results.

| Role | What it does |
|---|---|
| **Hardware** | Reads datasheets, register maps, and pinouts to answer hardware-level questions against the actual part. |
| **Firmware** | Writes and reviews C/C++ firmware, RTOS tasks, and driver code for the target platform. |
| **Debug** | Works through logs, error codes, and signal timing to find where a system is failing. |
| **Docs** | Turns finished work into documentation and handover notes. |
| **Vision** | Runs object detection and tracking for camera and sensor pipelines. |
| **Test / QA** | Sanity-checks generated code and answers before they reach the final output. |

The pipeline: **ingest → route → retrieve (RAG) → generate → verify & deliver.**

## Capabilities

- **Language & code** — firmware generation, code review, RTOS task design, protocol implementation, driver code
- **Knowledge & retrieval** — datasheet lookup, RAG over technical manuals, changelog writing
- **Debugging & diagnostics** — log analysis, error-code interpretation, root-cause narrowing
- **Vision & edge** — object detection (YOLO), on-device inference, sensor pipeline integration
- **Model engineering** — dataset curation, LoRA/QLoRA fine-tuning, GGUF conversion, deployment via llama.cpp

## Stack

`C` · `C++` · `Python` · `STM32` · `ESP32` · `nRF52` · `FreeRTOS` · `Zephyr` · `Embedded Linux` · `MQTT` · `TLS` · `YOLO` · `Local LLMs/SLMs` · `RAG` · `MCP` · `llama.cpp`

## Related work

- [`phi4-embedded-finetune`](./phi4-embedded-finetune) — fine-tuning Phi-4 14B on embedded systems C/C++ knowledge with LoRA, served locally via llama.cpp

## Contact

Independent work on embedded systems, IoT, and local AI — based in Izmir, Turkey.

[ugurakas@gmail.com](mailto:ugurakas@gmail.com) · [GitHub](https://github.com/ugurakas) · [LinkedIn](https://www.linkedin.com/in/ugur-akas/)

---
© 2026 UnitAI
