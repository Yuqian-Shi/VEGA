import copy
import glob
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import openai
import torch

from vega.evaluation_harness import image_utils
from vega.agent import construct_agent
from vega.browser_env import ScriptBrowserEnv
from vega.browser_env.env_config import reload_config # Check if we need to reload here
from vega.common.consts import SMY_FILE

from .utils import save_task_config, scan_results, get_unfinished_tasks, ConfigContainer
from .core import evaluate_single_task
from .models import generate_result_dir

def init_site_args(site_config, model_config, cfg, logger, new_result_dir, progress_queue=None):
    task_cfg = copy.deepcopy(cfg.config_data)
    task_cfg["model"] = copy.deepcopy(model_config)
    task_cfg["site"] = copy.deepcopy(site_config)
    if "sites" in task_cfg:
        del task_cfg["sites"]
    if "llm" in task_cfg:
        del task_cfg["llm"]
    task_cfg["site"]["result_dir"] = new_result_dir
    summary_path = os.path.join(new_result_dir, SMY_FILE)
    config_file_name = task_cfg.get("output", {}).get("config_file", "config.json")
    config_file = Path(new_result_dir) / config_file_name
    if config_file.exists():
        logger.info("Config file exists, skipping save.")
    else:
        save_task_config(task_cfg, logger)
    if "agent" in task_cfg and "planner_ip" in task_cfg["agent"]:
        task_cfg["model"]["planner_ip"] = task_cfg["agent"]["planner_ip"]

    site_browser_overrides = task_cfg.get("site", {}).get("browser_overrides")
    if isinstance(site_browser_overrides, dict):
        task_cfg.setdefault("browser", {}).update(site_browser_overrides)

    traces_dir = task_cfg.get("output", {}).get("traces_dir", "traces")
    (Path(new_result_dir) / traces_dir).mkdir(parents=True, exist_ok=True)
    log_file_path = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            log_file_path = handler.baseFilename
            break
    return task_cfg, summary_path, traces_dir, log_file_path

def evaluate_site_with_model(site_config, model_config, cfg, logger, result_dir, progress_queue=None):
    tmp_args = init_site_args(site_config, model_config, cfg, logger, result_dir, progress_queue)
    task_cfg, summary_path, traces_dir, log_file_path = tmp_args

    site_cfg = task_cfg.get("site")
    task_dir = site_cfg.get("task_dir")
    st_idx = site_cfg.get("start_idx", 0)
    ed_idx = site_cfg.get("end_idx")
    if ed_idx is None or ed_idx < st_idx:
        task_files = glob.glob(os.path.join(task_dir, "[0-9]*.json"))
        ed_idx = len(task_files)
        task_cfg["site"]["end_idx"] = ed_idx
        logger.info(
            f"end_idx set to None/ or smaller that start index, auto-detected {ed_idx} task files in {task_dir}"
        )

    job_jsons = []
    for i in range(st_idx, ed_idx):
        job_jsons.append(f"{task_dir}/{i}.json")
    
    sample_rate = site_cfg.get("sample_rate")
    if sample_rate is None:
        sample_rate = task_cfg.get("sample_rate", 1.0)
    
    if sample_rate < 1.0:
        total_available = len(job_jsons)
        k = max(1, int(total_available * sample_rate))
        rng = random.Random(42) 
        all_indices = list(range(total_available))
        rng.shuffle(all_indices)
        selected_indices = all_indices[:k]
        selected_indices.sort()
        job_jsons = [job_jsons[i] for i in selected_indices]
        logger.info(f"Sampled {len(job_jsons)}/{total_available} tasks (rate={sample_rate})")

    total_job_count = len(job_jsons)

    job_jsons = get_unfinished_tasks(job_jsons, result_dir)
    
    skipped_count = total_job_count - len(job_jsons)
    if progress_queue:
        for _ in range(skipped_count):
            progress_queue.put(1)

    if len(job_jsons) == 0:
        logger.info(f"No task left to run for site {site_cfg['site']}")
        return
    logger.info(f"Total {len(job_jsons)} tasks left for site {site_cfg['site']}")

    result = scan_results(result_dir)

    observation_type = task_cfg["model"].get("observation_type", task_cfg["model"].get("instruct_meta", {}).get("observation"))
    captioning_fn = None
    if observation_type in [
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        captioning_fn = image_utils.get_captioning_fn(
            device, dtype, task_cfg["model"]["captioning_model"]
        )

    agent = construct_agent(task_cfg, captioning_fn=captioning_fn, result_dir=result_dir)
    
    # Default to headless=True unless explicitly set to False in config.
    # Previously coupled with render_screenshot, but headless browsers can take screenshots too.
    is_headless = task_cfg["browser"].get("headless", True)
    
    blocked_resource_types = task_cfg["browser"].get("block_resources")
    if blocked_resource_types is None:
        blocked_resource_types = task_cfg["browser"].get("blocked_resource_types")

    sbe = ScriptBrowserEnv(
        headless=is_headless,
        slow_mo=task_cfg["browser"]["slow_mo"],
        observation_type=observation_type,
        current_viewport_only=task_cfg["browser"]["current_viewport_only"],
        viewport_size={
            "width": task_cfg["browser"]["viewport"]["width"],
            "height": task_cfg["browser"]["viewport"]["height"],
        },
        save_trace_enabled=task_cfg["browser"]["save_trace_enabled"],
        sleep_after_execution=task_cfg["browser"]["sleep_after_execution"],
        blocked_resource_types=blocked_resource_types,
        captioning_fn=captioning_fn,
    )
    for idx, job_json in enumerate(job_jsons):
        logger.info(f"Running task {idx + 1}/{len(job_jsons)}: {job_json}")
        task_id = os.path.basename(job_json).split(".")[0]

        was_error = False
        for err_key in result["error_info"]:
            if os.path.basename(err_key).split(".")[0] == task_id:
                was_error = True
                break

        try:
            ret, msg, timing = evaluate_single_task(
                task_cfg, job_json, sbe, agent, logger, result_dir=result_dir, captioning_fn=captioning_fn
            )
            
            if timing:
                for k in ["total_wall_time", "llm_thinking", "verify_model", "browser_action", "other"]:
                    if k in timing:
                        result["timing_stats"][k] += timing[k]
                result["timing_stats"]["count"] += 1

            if ret == "error":
                if not was_error:
                    result["error"] += 1
                if task_id not in result["error_info"]:
                    result["error_info"][task_id] = []
                result["error_info"][task_id].append(
                    {
                        "error_message": msg if msg else "Unknown Error",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "config": task_cfg,
                    }
                )
            elif ret == "correct":
                result["correct"] += 1
                if was_error:
                    result["error"] -= 1
            elif ret == "incorrect":
                result["incorrect"] += 1
                if was_error:
                    result["error"] -= 1
            elif ret == "partially_correct":
                result["partially_correct"] += 1
                if was_error:
                    result["error"] -= 1
        except openai.OpenAIError as e:
            logger.info(f"OpenAI API error during task {job_json}: {e}")
            if not was_error:
                result["error"] += 1
            if task_id not in result["error_info"]:
                result["error_info"][task_id] = []
            result["error_info"][task_id].append(
                {
                    "error_message": f"OpenAI API Error: {str(e)}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            print(e)
        except Exception as e:
            logger.exception(e)
            logger.error(f"Exception during task {job_json}: {e}")

            if not was_error:
                result["error"] += 1
            if task_id not in result["error_info"]:
                result["error_info"][task_id] = []
            result["error_info"][task_id].append(
                {
                    "error_message": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "config": task_cfg,
                }
            )
            error_file = task_cfg.get("output", {}).get("error_file", "error.txt")
            with open(
                Path(task_cfg["site"]["result_dir"]) / error_file,
                "a",
                encoding="utf-8",
            ) as f:
                f.write(f"[Job file]: {job_json}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                # f.write(traceback.format_exc()) 
            print(e)
        
        if progress_queue:
            progress_queue.put(1)

        # Refresh stats from disk to handle parallel workers
        # This prevents race conditions where starting with an old summary overwrites 
        # updates made by other workers in the meantime.
        latest_result = scan_results(result_dir)
        
        # Merge current task error info if any (local memory has most recent error state for this task)
        if task_id in result["error_info"]:
            if task_id not in latest_result["error_info"]:
                latest_result["error_info"][task_id] = result["error_info"][task_id]
                # If it wasn't known to disk, and we have an error, increment error count
                latest_result["error"] += 1
            else:
                latest_result["error_info"][task_id] = result["error_info"][task_id]
        
        latest_result["total"] = (
            latest_result["correct"]
            + latest_result["incorrect"]
            + latest_result["partially_correct"]
            + latest_result["error"]
        )
        
        result = latest_result 
        json.dump(result, open(summary_path, "w"), indent=4)
    sbe.close()
    logger.warning(
        f"Site {site_cfg['site']} testing completed. Summary: {json.dumps(result, indent=4)}"
    )
    json.dump(result, open(summary_path, "w"), indent=4)

def run_site_wrapper(args):
    site_config, model_config, cfg_data, log_file_base, progress_queue = args
    
    # Initialize globals in worker process
    reload_config(cfg_data)
    
    cfg = ConfigContainer(cfg_data)
    
    task_cfg = copy.deepcopy(cfg.config_data)
    task_cfg["model"] = copy.deepcopy(model_config)
    task_cfg["site"] = copy.deepcopy(site_config)
    
    if "sites" in task_cfg:
        del task_cfg["sites"]
    if "llm" in task_cfg:
        del task_cfg["llm"]

    original_result_dir = task_cfg["site"]["result_dir"]
    base_result_dir = (
        os.path.dirname(original_result_dir)
        if os.path.dirname(original_result_dir)
        else "results"
    )
    new_result_dir = generate_result_dir(task_cfg, base_result_dir)

    site_name = site_config['site']
    # If split, append range to logger name to avoid collision
    if site_config.get("_is_split"):
        st = site_config.get("start_idx", 0)
        ed = site_config.get("end_idx", "N")
        logger_name = f"logger_{site_name}_{st}_{ed}"
    else:
        logger_name = f"logger_{site_name}"
        
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
        
    log_file_name = os.path.basename(log_file_base).replace(".log", f"_{site_name}.log")
    # If split, append range to log file to avoid collision
    if site_config.get("_is_split"):
        st = site_config.get("start_idx", 0)
        ed = site_config.get("end_idx", "N") 
        log_file_name = log_file_name.replace(".log", f"_{st}_{ed}.log")
        
    log_file = os.path.join(new_result_dir, log_file_name)
    logger.info(f"Result directory: {new_result_dir}")
    fh = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter("%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    try:
        evaluate_site_with_model(site_config, model_config, cfg, logger,new_result_dir, progress_queue)
    except Exception as e:
        logger.exception(f"Failed to run site {site_name}: {e}")
