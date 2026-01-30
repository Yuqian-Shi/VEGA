# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Authors :       sundapeng.sdp
   Date：          2025/6/27
   Description :
-------------------------------------------------
"""
__author__ = 'sundapeng.sdp'

import json
from pathlib import Path
from typing import Dict, List, Union


def generate_webarena_task(
        task_id: int,
        intent: str,
        start_url: str,
        storage_state: str,
        intent_template: str,
        instantiation_dict: Dict[str, str],
        reference_answers: Dict[str, Union[str, List[str]]],
        reference_answer_raw_annotation: str,
        require_login: bool = True,
        require_reset: bool = False,
        intent_template_id: int = 0,
        geolocation: str = None,
        reference_url: str = "",
        string_note: str = "",
        program_html: List[str] = [],
        eval_types: str = "",
        *args, **kwargs
) -> Dict:
    """
    Generate a WebArena-lite style benchmark task dict.
    """
    if not intent:
        intent = intent_template
        for key, value in instantiation_dict.items():
            intent = intent.replace(f"{{{{{key}}}}}", value)

    task = {
        "task_id": task_id,
        "require_login": require_login,
        "storage_state": storage_state,
        "start_url": start_url,
        "geolocation": geolocation,
        "intent_template": intent_template,
        "instantiation_dict": instantiation_dict,
        "intent": intent,
        "require_reset": require_reset,
        "eval": {
            "eval_types": [eval_types],
            "reference_answers": reference_answers,
            "reference_url": reference_url,
            "program_html": program_html,
            "string_note": string_note,
            "reference_answer_raw_annotation": reference_answer_raw_annotation
        },
        "intent_template_id": intent_template_id
    }
    task.update(**kwargs)
    return task


if __name__ == "__main__":
    # Input Params
    website_tag = "openstack"
    task_metadata = {
        "sites": [
            "openstack_dashboard"
        ],
        "storage_state": "./.auth/openstack_state.json",
        "start_url": "http://127.0.0.1:8080",
    }

    start_idx = 40
    intent_and_ans_generated_file = "xx"
    output_file_type = "json"  # or jsonl

    base_path = Path.cwd()
    output_dir = base_path / "benchmark" / website_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(intent_and_ans_generated_file).exists():
        raise ValueError("Intent and answer generated file is not found.")

    with open(intent_and_ans_generated_file, "r") as f:
        list_intent_and_ans_generated = json.load(f)

    for idx, _ in enumerate(list_intent_and_ans_generated, start=start_idx):
        task = generate_webarena_task(
            task_id=idx,
            intent=_["intent"],
            require_reset=False,
            intent_template=_["intent_template"],
            instantiation_dict=_["instantiation_dict"],
            reference_answers=_["eval"]["reference_answers"],
            require_login=_["require_login"],
            reference_answer_raw_annotation=_["eval"]["reference_answer_raw_annotation"],
            intent_template_id=idx,
            **task_metadata
        )

        if output_file_type == "json":
            with open(output_dir / f"{idx}.json", "w") as f:
                json.dump(task, f, ensure_ascii=False, indent=4)
        elif output_file_type == "jsonl":
            with open(output_dir / f"{idx}.jsonl", "a") as f:
                json.dump(task, f, ensure_ascii=False, indent=4)
                f.write("\n")
