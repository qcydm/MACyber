# Stage 2: Continual Training & Gemma 3 Conversational Adaptation

This directory contains scripts and raw datasets used to adapt standard Alpaca formatted instruction datasets into structured conversational formats tailored for the Gemma 3 model.

## Directory Structure & Files

- **`process_gemma_data.py`**: Reads standard Alpaca-format JSON files from the root of this folder, parses their outputs, and converts them to:
  1. An updated instruction stating the new markdown JSON output format.
  2. A conversational output starting with a detailed, step-by-step natural language analysis (randomly formatted with 4 different investigation/assessment templates) followed by a clean markdown JSON block enclosing structured fields (`evidence`, `analysis`, `action`, `official`, `severity`).
- **`merge_json_lists.py`**: Discovers all domain-specific raw JSON datasets in this directory (excluding its own output), randomly samples exactly 200 records from each dataset to prevent bias, and merges them into a balanced subset `merged_random_data.json`.
- **`gemma_converted/`**: The output directory where Gemma 3 adapted files are stored.
- **`output_data_cleaned.json`**: Temporary output / sample target file.

## Available Domain Datasets

The raw JSON files present in this directory include:
- `CIC-BCCC-NRC2024.json`
- `CIC-BoT-IoT.json`
- `CIC-IDS-2017.json`
- `CIC-IoT-DIAD2024.json`
- `CIC-ToN-IoT.json`
- `CICEVSE2024.json`
- `CICIoMT 2024.json`
- `bccc_2024_exf_20251016_103047_train.json`
- `bccc_2024_mal_20251016_003244_train.json`
