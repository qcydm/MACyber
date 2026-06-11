# Stage 1: Pre-training & Fine-tuning Data Processing (Alpaca Format)

This directory contains the pipeline and source code for converting multi-source, heterogeneous cybersecurity logs into standard Alpaca format (`instruction`, `input`, `output`) dataset files.

## Directory Structure & Files

- **`alpaca_convert_v1.py`**: Reads V1 multi-source datasets (DNS, Log, Threat, Traffic, Vulnerability, etc.) and formats them into JSON matching the standard Alpaca instruction-following format.
- **`alpaca_convert_v2.py`**: The updated conversion script for the V2 dataset, incorporating a dataset validation logging mechanism and customized parameters (e.g. CUTOFF thresholds).
- **`alpaca_convert_legacy.py`**: Legacy data converter implementation.
- **`merge_json_lists.py`**: Combines multiple converted domain JSON lists into a single consolidated instruction training file, applying custom weights (repetition) on specific domains (e.g., traffic/threat/IDS datasets) to balance dataset domain distribution.
- **`analyze_attack_types.py`**: Parses official classification fields in processed datasets to generate a statistical summary of attack/incident type frequencies.
- **`token_count.py`**: Loads a tokenizer (default: Gemma 3) via the HuggingFace Mirror API to count dataset tokens (average and maximum lengths) and export reports.
- **`copy_to_llama_factory.py`**: Renames and copies processed stage 1 datasets to the `/data2/qcy/LLaMA-Factory/data` directory for training.
- **`url_convert.py` & `url_convert_meta.py`**: Specialized tools for importing Feodo-Tracker, ISCX-URL2016, and Malicious-URLs threat intel records.
- **`converted_file_paths.txt`**: Paths of successful V1 conversions.
- **`gemma_token_stats.txt`**: The generated Token count statistics report.

## Domain Categories Processed

1. **DNS**: Analyzing query logs for DGA, resolution anomalies, and tunneling.
2. **IoT**: Analyzing connection rates, flag anomalies, and protocols.
3. **Log**: Auditing Android/HDFS/Linux user commands and system traces.
4. **Threat**: Pulsedive and SABU Alert IOC reputation matches.
5. **Traffic**: Raw PCAP network flow feature assessment.
6. **URL**: Botnet C2, defacement, and malware domain hosting indicators.
7. **Vulnerability**: CVE vulnerability reports and exploit vector profiling.
