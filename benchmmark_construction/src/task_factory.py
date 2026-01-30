# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Authors :       sundapeng.sdp
   Date：          2025/7/9
   Description :
-------------------------------------------------
"""
__author__ = 'sundapeng.sdp'

import json
import os
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from helper.generate_webarena_task import generate_webarena_task as format_func


def require_field(cfg, key_path: str, error_msg=None, default=None):
    value = OmegaConf.select(cfg, key_path)
    if value is not None:
        return value
    elif default is not None:
        return default
    else:
        raise ValueError(error_msg or f"Missing required config key: {key_path}")


def load_playbook(playbook_dir, metadata_name="metadata.json", playbook_yaml_name="run_playbook.yaml",
                  playbook_info_name="scenario_description.json"):
    sub_dirs = [p for p in playbook_dir.iterdir() if p.is_dir()]
    print(sub_dirs)
    for dir in sub_dirs:
        hash_id = dir.name
        metadata_file = list(dir.rglob(metadata_name))
        playbook_yaml_file = list(dir.rglob(playbook_yaml_name))
        playbook_info_file = list(dir.rglob(playbook_info_name))
        yield (hash_id, metadata_file[0], playbook_yaml_file[0], playbook_info_file[0])


def load_raw_questions(raw_question_dir, raw_question_file):
    if raw_question_dir:
        raw_question_dir = Path(raw_question_dir)
        for file in raw_question_dir.rglob(raw_question_file):
            yield file
    else:
        yield Path(raw_question_file)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    output_base_dir = Path(cfg.output_base_dir)
    raw_question_dir = cfg.task_factory.raw_question_dir
    raw_question_file = cfg.task_factory.raw_question_file
    max_task_id = cfg.task_factory.max_task_id
    task_metadata_cfg = cfg.conf.task_metadata
    task_metadata = OmegaConf.to_container(task_metadata_cfg, resolve=True)

    if raw_question_dir and not Path(raw_question_dir).is_dir():
        raise ValueError("raw_question_dir must be a directory")

    if raw_question_file and not Path(raw_question_file).is_file():
        raise ValueError("raw_question_file must be a file")

    if not raw_question_dir and not raw_question_file:
        raise ValueError("Please specify either raw_question_dir or raw_question_file")

    task_base_dir = Path.joinpath(output_base_dir, "tasks_bank")
    task_base_dir.mkdir(parents=True, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    mapper_file = Path.joinpath(task_base_dir, "mapper.jsonl")
    tasks_bank_dir = Path.joinpath(task_base_dir, time_str)
    tasks_bank_dir.mkdir(parents=True, exist_ok=True)

    idx = max_task_id
    template_idx = 0
    intents = []
    question_files = []
    dict_intent_templates = {}
    count = 0
    for question_file in load_raw_questions(raw_question_dir, raw_question_file):
        question_files.append(str(question_file))
        with open(question_file, "r") as f:
            for q in f.readlines():
                question = json.loads(q)

                intent_template = question["template"]
                if intent_template not in dict_intent_templates:
                    dict_intent_templates.update({intent_template: template_idx})
                    template_idx += 1

                intent = question["question"]
                if intent in intents:
                    print(f"Spip question, intent: {intent}, Duplicate intent: {intent}")
                    continue

                answer = question["answer"]
                if "None" in answer:
                    print(f"Spip question, intent: {intent}, Invalid answer: {answer}")
                    continue

                task = format_func(
                    task_id=idx,
                    intent=intent,
                    zh_intent=question["zh_question"],
                    intent_template=intent_template,
                    instantiation_dict=question["placeholder_values"],
                    operation_type=question["operation_type"],
                    reference_answers={"fuzzy_match": question.get("answer")},
                    require_login=True,
                    require_reset=False if question["operation_type"] == "SELECT" else True,
                    reference_answer_raw_annotation=question.get("sql_execute_result"),
                    intent_template_id=dict_intent_templates.get(intent_template, ""),
                    eval_types="string_match",
                    verification_sql=question.get("verification_sql"),
                    **task_metadata,
                )
                print(json.dumps(task, indent=4, ensure_ascii=False))

                with open(Path.joinpath(tasks_bank_dir, f"{idx}.json"), "w") as f:
                    json.dump(task, f, ensure_ascii=False, indent=4)

                intents.append(intent)
                idx += 1
                count += 1

        with open(Path.joinpath(tasks_bank_dir, f"intent_templates.json"), "w") as f:
            json.dump(dict_intent_templates, f, ensure_ascii=False, indent=4)

    with open(mapper_file, "a") as f:
        json.dump({
            "task_bank_dir": time_str,
            "question_files": question_files,
            "sum_num": count
        }, f, ensure_ascii=False)
        f.write("\n")


if __name__ == '__main__':
    main()
