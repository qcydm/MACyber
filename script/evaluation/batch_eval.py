import os
import json
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "benchmark"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

MODEL_LIST = [
    "GPT-5.1",
    "GPT-5",
    "Claude4.5-Sonnet",
    "QWEN3-max",
    "Gemini3-pro",
    "SecGPT-14B",
    "Gemma3-27B",
    "GPT-oss-20B",
    "Llama4-17B",
    "DeepSeek-V3.2",
    "Qwen3-32B",
    "Mistral Small",
    "RedSage-Qwen3-8B-Base",
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BatchEval")

def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def get_unique_threat_types(json_path):
    """
    Extracts unique 'label.official' values from the ground truth JSON.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        threat_types = set()
        for item in data:
            # Handle both single-layer and nested 'json' structure if necessary
            # Based on convert_qwen.py, the label is in the root dictionary under 'label' 
            # OR inside 'json' -> 'label' depending on the stage.
            # convert_qwen output structure:
            # [ { "label": { "official": "..." }, ... } ]
            # Let's check the schema from convert_qwen.py output again.
            # It outputs: { "label": { "official": "..." }, "json": ..., "reasoning": ... }
            
            label_obj = item.get('label', {})
            if 'official' in label_obj:
                threat_types.add(label_obj['official'])
            elif 'json' in item and 'label' in item['json']:
                # Fallback if structure is nested differently
                threat_types.add(item['json']['label'].get('official'))
                
        return sorted(list(threat_types))
    except Exception as e:
        logger.error(f"Failed to extract threat types from {json_path}: {e}")
        return []

def iter_benchmark_files(data_dir, selected_datasets=None):
    selected = set(selected_datasets or [])
    for category_dir in sorted(data_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for input_json in sorted(category_dir.glob("*.json")):
            if input_json.name == "dataset_counts.csv":
                continue
            dataset_name = input_json.stem
            if selected and dataset_name not in selected:
                continue
            yield category_dir.name, dataset_name, input_json


def run_batch_evaluation(data_dir, results_root, model_name, workers=16, tiny=None, selected_datasets=None, api_key=None, base_url=None, api_model=None, judge_api_key=None, judge_model=None, use_rag=False, rag_db=None, rag_top_k=3):
    import generate_answers
    import evaluate_model

    if api_key is not None:
        generate_answers.API_KEY = api_key
    if base_url is not None:
        generate_answers.BASE_URL = base_url
    if api_model is not None:
        generate_answers.MODEL_NAME = api_model
    if judge_api_key is not None:
        evaluate_model.API_KEY = judge_api_key
    if judge_model is not None:
        evaluate_model.MODEL_NAME = judge_model

    data_path = Path(data_dir).resolve()
    results_path = Path(results_root).resolve() / model_name
    
    if not data_path.exists():
        logger.error(f"Benchmark data directory not found: {data_path}")
        return

    benchmark_files = list(iter_benchmark_files(data_path, selected_datasets))
    
    logger.info(f"Found {len(benchmark_files)} dataset files in {data_path}")
    
    for category_name, dataset_name, input_json in benchmark_files:
        logger.info(f"Processing dataset: {category_name}/{dataset_name}")
        
        dataset_result_dir = results_path / category_name / dataset_name
        dataset_result_dir.mkdir(parents=True, exist_ok=True)
        
        model_output_file = dataset_result_dir / "model_output.json"
        eval_result_file = dataset_result_dir / "eval_result.json"
        
        # # # 4. Run Generation
        logger.info(f"Starting generation for {dataset_name}...")
        try:
            generate_answers.generate_answers(
                input_file=str(input_json),
                output_file=str(model_output_file),
                dataset_name=dataset_name,
                max_workers=workers,
                tiny=tiny,
                use_rag=use_rag,
                rag_db_path=rag_db,
                rag_top_k=rag_top_k,
            )
        except Exception as e:
            logger.error(f"Generation failed for {dataset_name}: {e}")
            continue
            
        # 5. Run Evaluation
        logger.info(f"Starting evaluation for {dataset_name}...")
        try:
            evaluate_model.evaluate_model_output(
                standard_file=str(input_json),
                model_output_file=str(model_output_file),
                output_file=str(eval_result_file),
                max_workers=workers,
                tiny=tiny
            )
        except Exception as e:
            logger.error(f"Evaluation failed for {dataset_name}: {e}")
            continue
            
        logger.info(f"Completed {dataset_name}.\n")

def main():
    parser = argparse.ArgumentParser(description="Batch Evaluation Script")
    parser.add_argument('--data_dir', default=str(DEFAULT_DATA_DIR), help='MACyber data directory')
    parser.add_argument('--output_dir', default=str(DEFAULT_OUTPUT_DIR), help='Directory to store generated outputs and evaluation files')
    parser.add_argument('-m', '--model', default='QWEN3-max', help='Model name used as the result subdirectory')
    parser.add_argument('--datasets', nargs='*', default=None, help='Optional dataset names to evaluate')
    parser.add_argument('--api-key', default=None, help='OpenAI-compatible API key (默认读取 OPENAI_API_KEY)')
    parser.add_argument('--base-url', default=None, help='OpenAI-compatible base URL (默认读取 OPENAI_BASE_URL)')
    parser.add_argument('--api-model', default=None, help='OpenAI-compatible model name (默认读取 OPENAI_MODEL)')
    parser.add_argument('--judge-api-key', default=None, help='DashScope API key for evaluation (默认读取 DASHSCOPE_API_KEY)')
    parser.add_argument('--judge-model', default=None, help='DashScope judge model name (默认: qwen3-max)')
    parser.add_argument('--use-rag', action='store_true', help='Enable threat-intelligence RAG examples in generation prompts')
    parser.add_argument('--rag-db', default=None, help='Path to known_attack_RAG.json for the known-attack channel')
    parser.add_argument('--rag-top-k', type=int, default=3, help='Number of RAG examples to include')
    parser.add_argument('--workers', type=int, default=16, help='Number of worker threads')
    parser.add_argument('--tiny', type=int, default=None, help='Run on a small subset for testing')
    parser.add_argument(
        '--all',
        nargs='?',
        const=True,
        default=False,
        type=parse_bool,
        help='If true, run sequentially with every model name in MODEL_LIST as the -m/--model argument'
    )

    args = parser.parse_args()
    if args.all:
        for model_name in MODEL_LIST:
            logger.info("=" * 80)
            logger.info(f"Running batch evaluation for model: {model_name}")
            logger.info("=" * 80)
            run_batch_evaluation(
                args.data_dir,
                args.output_dir,
                model_name,
                args.workers,
                args.tiny,
                args.datasets,
                args.api_key,
                args.base_url,
                args.api_model,
                args.judge_api_key,
                args.judge_model,
                args.use_rag,
                args.rag_db,
                args.rag_top_k,
            )
    else:
        run_batch_evaluation(
            args.data_dir,
            args.output_dir,
            args.model,
            args.workers,
            args.tiny,
            args.datasets,
            args.api_key,
            args.base_url,
            args.api_model,
            args.judge_api_key,
            args.judge_model,
            args.use_rag,
            args.rag_db,
            args.rag_top_k,
        )

if __name__ == "__main__":
    main()
