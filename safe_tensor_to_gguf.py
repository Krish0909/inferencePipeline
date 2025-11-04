from pathlib import Path
from convert_hf_to_gguf import convert_model
import os

model_cache_dir = ".app/model/models--meta-llama--Llama-3.2-3B-Instruct/snapshots"
latest_snapshot = max(Path(model_cache_dir).iterdir(), key=os.path.getmtime)

output_gguf = "output_model.gguf"

# Call conversion
convert_model(str(latest_snapshot), outfile=output_gguf, outtype="q8_0")