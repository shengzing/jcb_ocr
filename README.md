<div align="center">
  <img src="images/logo.png" alt="JCB-OCR Logo" width="220" />
</div>

# JCB-OCR Hybrid OCR Analysis Suite

JCB-OCR is an interactive document-understanding platform that mixes layout-recognition and content-recognition models within a single workflow. Built with Streamlit, it ships with the `local_vl_utils` toolkit—unified clients, prompt templates, and post-processing helpers for multiple VLM providers—so you can run hybrid OCR pipelines locally or against remote APIs with minimal effort.


<p align="center">
  <a href="README_CN.md"><img src="https://img.shields.io/badge/Docs-中文指南-blue.svg?style=flat"></a>
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-1.x-brightgreen.svg?style=flat"></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10+-yellow.svg?style=flat"></a>
</p>

## Project Background
- **Hybrid necessity**: A single OCR model rarely excels at layout analysis, table/formula understanding, and high-quality text extraction simultaneously. JCB-OCR offers an extensible architecture that stitches different capabilities together through a consistent interface.
- **Rapid validation & comparison**: Interactive test/comparison panels let you switch configurations, observe logs, and save multiple runs without leaving the UI, dramatically shortening verification cycles.
- **Standardized outputs**: Built-in converters normalize diverse model outputs into a single JSON schema (plus visual previews), reducing integration effort for downstream business systems.

## Key Features
- **Decoupled model pairing**: Combine Qwen-VL, DeepSeek-VL, Hunyuan-VL, MinerU, or custom clients independently for layout vs. content extraction.
- **Unified JSON representation**: `format_converter` plus post-processors transform raw outputs into a shared structure (`blocks + metadata`) ready for ingestion.
- **Visual & comparative analysis**: Inspect logs, JSON, and overlays in real time; leverage comparison tabs to contrast configurations on identical inputs.
- **Batch & async pipelines**: `local_vl_utils` wraps HTTP and vLLM clients with retry/async support, enabling large-scale processing or low-latency applications.
- **Traceable logging**: API requests, diagnostics, and downloadable reports are built-in for troubleshooting and auditing.

## Feature Overview
- **Interactive workbench**: Upload images/PDFs, choose model combos instantly, watch logs, and export JSON or visualized results with a click.
- **Recognition statistics dashboard**: Automatically summarizes block counts, averages, and per-type distributions, with charts to monitor coverage and accuracy.
- **Comparison mode**: Preserve multiple runs, generate Spans PDFs, and view side-by-side metrics to identify the best-performing configuration quickly.
- **Unified output adapters**: Ensure every model’s layout/content data conforms to the same schema, simplifying downstream integration.
- **Pluggable client system**: Register custom OCR/VLM clients via `client_factory`, configured through `.env` or runtime parameters.
- **Rich post-processing chain**: Table reconstruction, equation fixes, handwriting handling, and more can be composed as needed.
- **API/log monitoring**: HTTP clients capture detailed request/response traces and store them under `logs/api_calls/` for postmortem analysis.

## Supported Block Types
- **Text**: Paragraphs, headings, lists, etc., with precise coordinates.
- **Tables**: Detects structures, merged cells, and exports JSON/HTML.
- **Images & seals**: Distinguishes pictures, stamps, and seals for targeted post-processing.
- **Equations**: Multiple LaTeX fixers address double subscripts, left/right mismatches, etc.
- **Handwriting**: Dedicated block type for signatures and handwritten notes.

All types can be previewed visually or consumed via the standardized JSON output.

## UI Snapshots
<div align="center">
  <img src="images/image.png" alt="Model selection & upload" width="95%" />
  <p>Main panel: model selection and file upload</p>
</div>
<div align="center">
  <img src="images/image1.png" alt="Processing logs" width="95%" />
  <p>Real-time processing logs</p>
</div>
<div align="center">
  <img src="images/image2.png" alt="JSON results" width="95%" />
  <p>Unified JSON results with download options</p>
</div>
<div align="center">
  <img src="images/image3.png" alt="Visual preview" width="95%" />
  <p>Visual overlays of detected blocks</p>
</div>
<div align="center">
  <img src="images/image5.png" alt="Statistics dashboard" width="95%" />
  <p>Recognition statistics and type distributions</p>
</div>

## Directory Layout
```
.
├─ app/                  # Streamlit interface
│  ├─ streamlit_app.py   # UI + workflow logic
│  └─ requirements.txt   # App dependencies
└─ local_vl_utils/       # Client toolkit & helpers
   ├─ base_client.py
   ├─ client_factory.py
   ├─ config_loader.py
   ├─ prompt_library.py
   ├─ vlm_client/        # HTTP/vLLM implementations
   ├─ post_process/      # Post-processing modules
   └─ examples/          # JSON conversion demos
```

## Environment Setup
1. Install Python 3.10+ (virtualenv recommended).
2. Install dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```
3. Configure model credentials in `vlm_client/.env` (or the project `.env`):
   ```env
   QWEN_MODEL_NAME=Qwen2-VL-72B-Instruct
   QWEN_SERVER_URL=https://api.example.com
   QWEN_API_KEY=sk-xxx
   DEFAULT_TIMEOUT=600
   DEFAULT_MAX_RETRIES=3
   ```
   Repeat for DEEPSEEK/HUNYUAN/MINERU with their prefixes; each supports its own timeout/retry overrides.

## Running the App
1. Ensure `.env` is populated and target endpoints are reachable.
2. Launch Streamlit from the repo root:
   ```bash
   streamlit run app/streamlit_app.py
   ```
3. In the UI:
   - Pick layout/content models from the sidebar.
   - Upload images or PDFs; enable “Comparison Mode” to store multiple runs.
   - Review logs, JSON output, and visual previews.
   - Generate Spans PDFs from the comparison tab when needed.

## Customization & Extension
- **Client registration**: Plug in custom OCR/VLM backends via `register_custom_client`.
- **Batch/async flows**: `BaseOCRClient.aio_batch_predict` lets you script large-scale or concurrent jobs.
- **Post-processing pipeline**: Extend or mix modules under `post_process` for domain-specific needs.
- **Logging**: API call logs live in `logs/api_calls/`, and the UI exposes download buttons for audits.

## Contributing
We welcome contributions! Before submitting:
1. **Follow the style guide**: use `loguru` for logging, avoid stray prints, and document non-trivial code.
2. **Branch & commit cleanly**: isolate each feature/fix and use descriptive messages (e.g., `feat: add mineru helper`).
3. **Validate changes**: run `streamlit run app/streamlit_app.py` (or equivalent tests) and mention results in your PR.
4. **Sync dependencies**: update the appropriate `requirements` file for any new libraries and ensure compatibility.

Open an Issue to discuss ideas or jump straight into a Pull Request—let’s build a robust document-understanding framework together.

## License & Attribution
This project uses the **JCB-OCR Attribution License 1.0** (`LICENSE`). Any use, modification, or redistribution must retain the copyright notice and include a visible attribution such as “Derived from JCB-OCR” with a link to the repository.

For commercial collaborations or additional licensing terms, please contact the maintainers.
