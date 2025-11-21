import os
import time
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from inferencePipeline import loadPipeline

# Environment settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=str, default=None, help='Track experiment (saves to experiments/{name}.json)')
    parser.add_argument('--metadata', type=str, default=None, help='Metadata as key=value,key=value')
    args = parser.parse_args()
    
    # Config
    input_file = "sample_questions.xlsx"   
    output_file = "answers_output.csv"

    print("[MAIN] Starting pipeline run...")
    
    # Load data
    df = pd.read_excel(input_file)
    questions = [{"questionID": row["questionID"], "question": row["question"]} for _, row in df.iterrows()]
    print(f"[MAIN] Loaded {len(questions)} questions")

    # Tracking setup
    if args.track:
        from inferencePipeline.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(args.track)
        tracker.__enter__()
        
        if args.metadata:
            metadata = {}
            for pair in args.metadata.split(','):
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    metadata[k.strip()] = v.strip()
            tracker.log_metadata(metadata)
        
        pipeline = tracker.track(loadPipeline())
    else:
        pipeline = loadPipeline()

    # Run inference
    start_time = time.time()
    answers = pipeline(questions)
    print(f"[MAIN] Done in {time.time() - start_time:.2f}s")

    # Close tracker
    if args.track:
        tracker.__exit__(None, None, None)

    # Save results
    answers_df = pd.DataFrame(answers)
    merged_df = df.merge(answers_df, on="questionID", how="left")
    merged_df.to_csv(output_file, index=False)
    print(f"[MAIN] Saved to {output_file}")
