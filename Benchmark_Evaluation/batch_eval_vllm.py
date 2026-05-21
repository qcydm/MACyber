import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import signal

import logging
import argparse
from pathlib import Path


def cleanup_gpu_and_exit(signal_number, frame):
    try:
        import torch
        torch.cuda.empty_cache()
    finally:
        exit(0)


signal.signal(signal.SIGTERM, cleanup_gpu_and_exit)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = REPO_ROOT / "MACyber-INT_benchmark"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

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


def run_batch_evaluation(data_dir, results_root, model_name, model_dir, workers=16, gpu_memory_utilization=0.1, tiny=None, selected_datasets=None, judge_api_key=None, judge_model=None, use_rag=False, rag_db=None, rag_top_k=3):
    import generate_answers_vllm
    import evaluate_model

    if judge_api_key is not None:
        evaluate_model.API_KEY = judge_api_key
    if judge_model is not None:
        evaluate_model.MODEL_NAME = judge_model

    data_path = Path(data_dir).resolve()
    results_path = Path(results_root).resolve() / model_name
    model_path = Path(model_dir).resolve()
    
    if not data_path.exists():
        logger.error(f"Benchmark data directory not found: {data_path}")
        return
    if not model_path.exists():
        logger.error(f"Model directory not found: {model_path}")
        return

    benchmark_files = list(iter_benchmark_files(data_path, selected_datasets))
    
    logger.info(f"Found {len(benchmark_files)} dataset files in {data_path}")
    
    for category_name, dataset_name, input_json in benchmark_files:
        logger.info(f"Processing dataset: {category_name}/{dataset_name}")
        
        dataset_result_dir = results_path / category_name / dataset_name
        dataset_result_dir.mkdir(parents=True, exist_ok=True)
        
        model_output_file = dataset_result_dir / "model_output.json"
        eval_result_file = dataset_result_dir / "eval_result.json"
        
        # 2. Extract Threat Types dynamically
        # threat_types = get_unique_threat_types(input_json)
        # if not threat_types:
        #     logger.warning(f"No threat types found for {dataset_name}, using default ['benign', 'malicious']")
        #     threat_types = ['benign', 'malicious']
        # else:
        #     logger.info(f"Extracted threat types for {dataset_name}: {threat_types}")

        # # 3. Patch generate_answers with the threat types
        # generate_answers.DATASET_THREAT_TYPES[dataset_name] = threat_types
        
        # 4. Run Generation
        logger.info(f"Starting generation for {dataset_name}...")
        try:
            generate_answers_vllm.generate_answers(
                input_file=str(input_json),
                output_file=str(model_output_file),
                dataset_name=dataset_name,
                model_path=str(model_path),
                GPU_MEMORY_UTILIZATION=gpu_memory_utilization,
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
    parser.add_argument('-m', '--model', default='vllm-model', help='Model name used as the result subdirectory')
    parser.add_argument('--model_dir', default="", help='directory of model')
    parser.add_argument('--datasets', nargs='*', default=None, help='Optional dataset names to evaluate')
    parser.add_argument('--judge-api-key', default=None, help='DashScope API key for evaluation (默认读取 DASHSCOPE_API_KEY)')
    parser.add_argument('--judge-model', default=None, help='DashScope judge model name (默认: qwen3-max)')
    parser.add_argument('--use-rag', action='store_true', help='Enable threat-intelligence RAG examples in generation prompts')
    parser.add_argument('--rag-db', default=None, help='Path to known_attack_RAG.json for the known-attack channel')
    parser.add_argument('--rag-top-k', type=int, default=3, help='Number of RAG examples to include')
    parser.add_argument('--workers', type=int, default=16, help='Number of worker threads')
    parser.add_argument('--gpu', type=float, default=0.1, help='GPU memory utilization')
    parser.add_argument('--tiny', type=int, default=None, help='Run on a small subset for testing')
    
    args = parser.parse_args()
    
    run_batch_evaluation(
        args.data_dir,
        args.output_dir,
        args.model,
        args.model_dir,
        args.workers,
        args.gpu,
        args.tiny,
        args.datasets,
        args.judge_api_key,
        args.judge_model,
        args.use_rag,
        args.rag_db,
        args.rag_top_k,
    )

if __name__ == "__main__":
    main()
