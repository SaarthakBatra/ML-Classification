import argparse
import os
import sys


RESULT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RESULT_DIR, "blocking", "src"))

from workflow import BlockingWorkflowConfig, run_blocking_workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--per-stream-batch-size", type=int, default=20)
    parser.add_argument("--max-seen-candidates", type=int, default=60)
    arguments = parser.parse_args()
    summary = run_blocking_workflow(
        BlockingWorkflowConfig(
            preprocessed_dir=arguments.preprocessed_dir,
            output_dir=arguments.output_dir,
            model_name=arguments.model_name,
            batch_size=arguments.batch_size,
            device=arguments.device,
            per_stream_batch_size=arguments.per_stream_batch_size,
            max_seen_candidates=arguments.max_seen_candidates,
        )
    )
    print(f"Wrote {summary['total_candidates']:,} candidate pairs to {summary['candidate_pairs_path']}")
    print(summary["classifier_status"])


if __name__ == "__main__":
    main()
