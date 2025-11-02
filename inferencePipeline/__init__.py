"""
EXTREME Performance LLM Pipeline - Maximum Speed Optimizations
Target: Sub-18 seconds with maintained accuracy
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import gc

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# Disable torch warnings for speed
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

class OptimizedInferencePipeline:
    def __init__(self):
        # Maximum CPU utilization
        torch.set_num_threads(16)
        torch.set_num_interop_threads(1)
        
        # Disable gradient computation globally
        torch.set_grad_enabled(False)
        
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.batch_size = 28  # Increased from 24
        
        # Pre-compute common prompt templates to avoid string operations
        self.prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
    def load_model(self):
        """Load model with extreme optimizations"""
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        cache_dir = './app/model'
        
        # Load tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=True,
            padding_side='left',
            use_fast=True  # Use fast tokenizer for speed
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load with bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            device_map={"":"cpu"},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",  # Faster for CPU inference
        )
        
        # Set to eval and disable dropout for consistency
        self.model.eval()
        for module in self.model.modules():
            if hasattr(module, 'dropout'):
                module.dropout = 0.0
        
        # Torch compile with aggressive settings
        try:
            self.model = torch.compile(
                self.model, 
                mode="max-autotune",  # Most aggressive optimization
                fullgraph=True
            )
        except:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except:
                pass
        
    def create_prompt(self, question: str) -> tuple:
        """
        Ultra-fast prompt creation with pre-computed templates.
        Returns (prompt, max_tokens)
        """
        ql = question.lower()
        
        # Fast keyword lookup with early exit
        # Check most common first (geography)
        if 'capital' in ql or 'country' in ql or 'city' in ql or 'continent' in ql or 'geography' in ql or \
           'ocean' in ql or 'mountain' in ql or 'river' in ql or 'located' in ql or 'border' in ql or \
           'lake' in ql or 'sea' in ql or 'island' in ql or 'desert' in ql or 'largest' in ql or \
           'highest' in ql or 'deepest' in ql or 'longest' in ql or 'known as' in ql or 'called' in ql or \
           'land of' in ql or 'region' in ql or 'territory' in ql or 'nation' in ql:
            system_msg = "You are a geography expert. Provide accurate, concise answers about locations, countries, and geography. When answering about nicknames or cultural references (like 'God's Own Country', 'Land of Rising Sun'), clearly state which place it refers to and provide brief context."
            return self.prompt_template.format(system_msg=system_msg, question=question), 125
        
        # Math - second most common
        if 'solve' in ql or 'calculate' in ql or 'equation' in ql or 'x=' in ql or 'algebra' in ql or \
           'factor' in ql or 'simplify' in ql or 'evaluate' in ql or 'compute' in ql or 'polynomial' in ql or \
           '+' in ql or '-' in ql or '*' in ql or '/' in ql:
            system_msg = "You are a math expert. Solve the problem step by step and provide the final answer clearly."
            return self.prompt_template.format(system_msg=system_msg, question=question), 155
        
        # History
        if 'when' in ql or 'who' in ql or 'history' in ql or 'war' in ql or 'century' in ql or \
           'year' in ql or 'historical' in ql or 'ancient' in ql or 'empire' in ql or 'battle' in ql or \
           'king' in ql or 'queen' in ql or 'president' in ql or 'dynasty' in ql or 'revolution' in ql or \
           'treaty' in ql or 'founded' in ql or 'established' in ql or 'reign' in ql or 'born' in ql or 'died' in ql:
            system_msg = "You are a history expert. Provide accurate, concise answers about historical events, dates, and figures. CRITICAL: Always include specific YEARS (and month/day if known). If asked 'when', begin your answer with the exact date. Use full proper names. Example: 'The French Revolution began in 1789.' Be factual and precise."
            return self.prompt_template.format(system_msg=system_msg, question=question), 185
        
        # General fallback
        system_msg = "You are a helpful educational assistant. Provide clear, accurate, and concise answers."
        return self.prompt_template.format(system_msg=system_msg, question=question), 135
    
    def process_batch(self, questions: List[str]) -> List[str]:
        """Extreme-speed batch processing"""
        
        # Batch create prompts
        prompt_data = [self.create_prompt(q) for q in questions]
        prompts = [p for p, _ in prompt_data]
        max_tokens = max(t for _, t in prompt_data)
        
        # Ultra-fast tokenization
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=320,  # Reduced from 350
            return_attention_mask=True
        ).to(self.device)
        
        # Fastest generation settings
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_tokens,
            min_new_tokens=5,  # Reduced from 8
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            repetition_penalty=1.02,  # Reduced from 1.03
            no_repeat_ngram_size=3,
        )
        
        # Vectorized decoding
        answers = []
        for i in range(len(outputs)):
            generated_tokens = outputs[i][inputs['input_ids'][i].shape[0]:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Minimal cleanup
            if answer.startswith("assistant"):
                answer = answer[9:].lstrip("\n").strip()
            
            if len(answer) < 3:
                answer = "Unable to generate answer."
            elif len(answer) > 5000:
                answer = answer[:5000]
            
            answers.append(answer)
        
        return answers
    
    def __call__(self, questions: List[Dict]) -> List[Dict]:
        """Main inference - maximum performance"""
        
        if self.model is None:
            self.load_model()
        
        question_texts = [q['question'] for q in questions]
        
        # Sort by length for optimal batching
        indexed_questions = [(i, q) for i, q in enumerate(question_texts)]
        indexed_questions.sort(key=lambda x: len(x[1]))
        
        all_answers = [None] * len(questions)  # Pre-allocate
        
        # Process in maximum batch sizes
        for i in range(0, len(indexed_questions), self.batch_size):
            batch_end = min(i + self.batch_size, len(indexed_questions))
            batch = indexed_questions[i:batch_end]
            
            batch_indices = [idx for idx, _ in batch]
            batch_qs = [q for _, q in batch]
            
            try:
                batch_answers = self.process_batch(batch_qs)
                
                # Direct assignment to pre-allocated array
                for orig_idx, ans in zip(batch_indices, batch_answers):
                    all_answers[orig_idx] = ans
                
            except Exception as e:
                for orig_idx in batch_indices:
                    all_answers[orig_idx] = "Unable to generate answer."
            
            # Even less frequent GC
            if i > 0 and i % 280 == 0:
                gc.collect()
        
        # Format output (single pass)
        return [
            {
                "questionID": questions[i]['questionID'],
                "answer": all_answers[i] if all_answers[i] else "Unable to generate answer."
            }
            for i in range(len(questions))
        ]


def loadPipeline():
    """Factory function to create and return the inference pipeline."""
    pipeline = OptimizedInferencePipeline()
    return pipeline