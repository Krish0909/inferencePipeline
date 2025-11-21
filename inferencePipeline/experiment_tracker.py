"""
Simple experiment tracking.
Usage: python load.py --track experiment_name
"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List


class ExperimentTracker:
    """Simple experiment tracker."""
    
    def __init__(self, name: str, output_dir: str = "./experiments"):
        self.name = name
        self.output_dir = Path(output_dir)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.metadata: Dict = {}
        self._pipeline = None
        self._original_call = None
        self.question_times: List[float] = []
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        total_time = self.end_time - self.start_time
        
        # Restore model
        if self._pipeline and self._original_call:
            self._pipeline.model.__call__ = self._original_call
        
        # Save results
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"{self.name}_{timestamp}.json"
        
        num_questions = len(self.question_times)
        results = {
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'metadata': self.metadata,
            'total_time_sec': round(total_time, 2),
            'total_questions': num_questions,
            'avg_latency_ms': round((total_time / num_questions) * 1000, 1) if num_questions > 0 else 0,
            'questions_per_sec': round(num_questions / total_time, 2) if total_time > 0 else 0
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"[TRACKER] Saved: {filepath}")
    
    def track(self, pipeline):
        """Wrap pipeline to track timing."""
        self._pipeline = pipeline
        self._original_call = pipeline.model.__call__
        tracker = self
        
        def timed_call(prompt, **kwargs):
            t0 = time.perf_counter()
            result = tracker._original_call(prompt, **kwargs)
            tracker.question_times.append((time.perf_counter() - t0) * 1000)
            return result
        
        pipeline.model.__call__ = timed_call
        return pipeline
    
    def log_metadata(self, metadata: dict):
        """Add metadata."""
        self.metadata.update(metadata)
