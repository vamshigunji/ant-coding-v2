"""
Adapter for loading tasks from the SWE-bench dataset.
"""

import logging
from typing import List, Optional
from ant_coding.tasks.types import Task, TaskSource, TaskDifficulty

logger = logging.getLogger(__name__)

def load_swebench(
    split: str = "lite", 
    limit: Optional[int] = None,
    subset: str = "verified"
) -> List[Task]:
    """
    Load tasks from SWE-bench.
    """
    try:
        import datasets
    except ImportError:
        raise ImportError("Please install datasets to use SWE-bench: pip install datasets")

    logger.info(f"Loading SWE-bench dataset: split={split}, subset={subset}")
    
    # Dataset name on Hugging Face: princeton-nlp/SWE-bench_Lite, etc.
    dataset_name = f"princeton-nlp/SWE-bench_{split.capitalize()}"
    if split.lower() == "lite":
        dataset_name = "princeton-nlp/SWE-bench_Lite"
    elif split.lower() == "verified":
        dataset_name = "princeton-nlp/SWE-bench_Verified"
    
    try:
        dataset = datasets.load_dataset(dataset_name, split="test")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        return []

    tasks = []
    count = 0
    
    for item in dataset:
        if limit and count >= limit:
            break
            
        # SWE-bench uses instance_id as the unique identifier
        task = Task(
            id=item["instance_id"],
            description=item["problem_statement"],
            source=TaskSource.SWE_BENCH,
            difficulty=TaskDifficulty.HARD, # SWE-bench tasks are generally hard
            max_tokens_budget=200_000,
            timeout_seconds=3600,
            metadata={
                "repo": item["repo"],
                "base_commit": item["base_commit"],
                "version": item["version"],
                "test_patch": item["test_patch"],
                "repo_url": f"https://github.com/{item['repo']}"
            }
        )
        tasks.append(task)
        count += 1
        
    return tasks
