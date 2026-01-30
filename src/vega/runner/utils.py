import os
import json
import shutil
import datetime
import logging
import sys
import time
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw
import requests

from vega.browser_env import (
    Trajectory,
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    create_stop_action,
)
from vega.browser_env.helper_functions import get_action_description
from vega.browser_env.actions import is_equivalent


class ConfigContainer:
    def __init__(self, data):
        self.config_data = data


def check_early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int], actions=None
) -> tuple[bool, str]:
    """Check whether need to stop early"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [action["action_type"] == ActionTypes.NONE for action in last_k_actions]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    if actions is None:
        last_action: Action = action_seq[-1]
        if last_action["action_type"] != ActionTypes.TYPE:
            if len(last_k_actions) >= k:
                if all(
                    [is_equivalent(action, last_action) for action in last_k_actions]
                ):
                    return True, f"Same action for {k} times"
        else:
            # check the action sequence
            if sum([is_equivalent(action, last_action) for action in action_seq]) >= k:
                return True, f"Same typing action for {k} times"
        return False, ""

    else:
        last_k_actions = actions[-k:]
        last_action = actions[-1]
        if len(last_k_actions) >= k:
            if all([action == last_action for action in last_k_actions]):
                return True, f"Same action for {k} times"
        return False, ""


def save_action_history(
    path: str, task_id: int, actions: List[str], score: float = -0.1, final_answer: str = None
):
    """Update action history file with proper file handling."""
    obj = {"task_id": task_id, "score": score, "actions": actions}
    if final_answer is not None:
        obj["final_answer"] = final_answer
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Use context manager to properly close file
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def execute_action_step(
    action,
    sbe,
    task_cfg,
    result_dir,
    task_id,
    actions,
    meta_data,
    render_helper,
    state_info,
    logger,
    agent,
):
    """Process the action: render, screenshot, and update metadata."""
    # Import PromptAgent here to avoid potential circular imports
    from vega.agent import PromptAgent

    enable_screenshot = task_cfg["browser"]["render_screenshot"]

    action_str = get_action_description(
        action,
        state_info["info"]["observation_metadata"],
        action_set_tag=task_cfg["model"].get("action_type", task_cfg["model"].get("instruct_meta", {}).get("action_type")),
        prompt_constructor=(
            agent.prompt_constructor if isinstance(agent, PromptAgent) else None
        ),
    )

    current_screenshot = None
    if enable_screenshot:
        current_screenshot = os.path.join(
            result_dir, "screenshots", f"{task_id}", f"{len(actions)}.png"
        )
        
        element_id = action["element_id"]
        should_annotate = False
        bbox = None
        
        if element_id != "":
            element = sbe.page.query_selector(f"[data-label-id='{element_id}']")
            if element:
                bbox_rect = element.bounding_box()
                if bbox_rect:
                    bbox = [
                        int(bbox_rect["x"]),
                        int(bbox_rect["y"]),
                        int(bbox_rect["width"]),
                        int(bbox_rect["height"]),
                    ]
                    should_annotate = True

        try:
            if should_annotate:
                # Get bytes, decode, draw, save - avoids double disk I/O
                screenshot_bytes = sbe.page.screenshot()
                nparr = np.frombuffer(screenshot_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                cv2.rectangle(
                    image,
                    (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    (0, 255, 0),
                    2,
                )
                cv2.circle(
                    image,
                    (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)),
                    radius=0,
                    color=(0, 255, 0),
                    thickness=2,
                )
                cv2.imwrite(current_screenshot, image)
            else:
                # Direct save is faster if no annotation needed
                sbe.page.screenshot(path=current_screenshot)
        except Exception as e:
            logger.warning(f"Failed to take screenshot: {e}")

    render_helper.render(action, state_info, meta_data, enable_screenshot, screenshot_path=current_screenshot)

    meta_data["action_history"].append(action_str)
    actions.append(action_str)
    print("Action String: ", action_str)

    action_type = action["action_type"]
    if action_type == ActionTypes.CLICK:
        # Click at relative coordinates (left, top) and save a screenshot with the click point.
        viewport_size = sbe.page.viewport_size
        assert viewport_size is not None
        bop = sbe.observation_handler.action_processor
        element_id = action["element_id"]
        assert type(element_id) in [str], f"Invalid element_id type: {type(element_id)}"
        
        # Log available element IDs before accessing
        try:
            element_center = (bop.get_element_center(element_id))
            left = element_center[0]
            top = element_center[1]
            x = int(left * viewport_size["width"])
            y = int(top * viewport_size["height"])
            if bop.observation_type == "image_som":
                bbox_left, bbox_top, width, height = bop.som_id_info[element_id]
                bbox_left, bbox_top, bbox_right, bbox_bottom = (
                    bbox_left,
                    bbox_top,
                    bbox_left + width,
                    bbox_top + height,
                )
            else:
                node_info = bop.obs_nodes_info[element_id]
                bbox = node_info.get(
                    "union_bound"
                )
                bbox_left, bbox_top, bbox_right, bbox_bottom = (
                    bbox[0],
                    bbox[1],
                    bbox[0] + bbox[2],
                    bbox[1] + bbox[3],
                )
            if enable_screenshot:
                img = Image.open(current_screenshot)
                draw = ImageDraw.Draw(img)
                r = 10
                draw.ellipse(
                    (x - r, y - r, x + r, y + r), fill="red", outline="black"
                )
                draw.rectangle(
                    (bbox_left, bbox_top, bbox_right, bbox_bottom),
                    outline="green",
                    width=3,
                )
                base_path = Path(
                    os.path.join(result_dir, "screenshots", f"{task_id}")
                )
                datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                os.makedirs(base_path, exist_ok=True)
                save_path = os.path.join(
                    base_path, f"click_point_{datetime_str}.png"
                )
                img.save(save_path)
                print(f"Click point marked and saved to: {save_path}")
        except Exception as e:
            logger.exception(e)
            logger.warning(f"Visualization failed for element {element_id}(type:{type(element_id)}) : {e}")
            # logger.info(f"Current Observation Context: {state_info.get('observation', 'No observation available')}")


def load_task_images(image_paths):
    images = []
    if image_paths is not None:
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        for image_path in image_paths:
            if image_path.startswith("http"):
                input_image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                input_image = Image.open(image_path)
            images.append(input_image)
    return images


def setup_task_directories(task_cfg, task_id):
    result_dir = task_cfg["site"]["result_dir"]
    out_path = os.path.join(result_dir, "actions", f"{task_id}.json")
    screenshot_dir = os.path.join(result_dir, "screenshots", f"{task_id}")
    if os.path.exists(screenshot_dir):
        shutil.rmtree(screenshot_dir)
    os.makedirs(screenshot_dir)
    return result_dir, out_path


def save_webrl_traces(sbe, trajectory, result_dir, task_id, intent, traces_dir="traces"):
    boh = sbe.observation_handler
    if boh.action_processor.observation_type == "webrl":
        current_path = os.path.join(result_dir, traces_dir, f"{task_id}.jsonl")
        traces = []
        for i in range(1, len(trajectory), 2):
            action = trajectory[i]
            state_info = trajectory[i - 1]
            obs = state_info["observation"]["text"]
            action_str = action["raw_prediction"]
            item = {
                "trace_id": task_id,
                "index": i // 2,
                "prompt": intent if i == 1 else "** Simplified html **",
                "html": obs,
                "response": action_str,
                "target": intent,
            }
            traces.append(item)
        with open(current_path, "w", encoding="utf-8") as f:
            for item in traces:
                f.write(json.dumps(item) + "\n")

def get_unfinished_tasks(config_files: list[str], result_dir: str) -> list[str]:
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        action_file = os.path.join(result_dir, "actions", f"{task_id}.json")
        
        is_finished = False
        if os.path.exists(action_file):
            try:
                with open(action_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Check if score is valid (not -0.1)
                if data.get("score", -0.1) != -0.1:
                    is_finished = True
            except Exception as e:
                print(f"Error reading action file {action_file}: {e}")
                pass
        
        if not is_finished:
            unfinished_configs.append(config_file)
            
    return unfinished_configs


def save_task_config(cfg, logger) -> None:
    """Save configuration to result directory."""
    config_file_name = cfg.get("output", {}).get("config_file", "config.json")
    config_file = Path(cfg["site"]["result_dir"]) / config_file_name
    if not config_file.exists():
        # Create a serializable dict
        config_dict = cfg
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=4)
            logger.info(f"Dump config to {config_file}")

def scan_results(result_dir):
    result = {
        "total": 0,
        "error": 0,
        "correct": 0,
        "incorrect": 0,
        "partially_correct": 0,
        "error_info": {},
        "timing_stats": {
            "total_wall_time": 0.0,
            "llm_thinking": 0.0,
            "verify_model": 0.0,
            "browser_action": 0.0,
            "other": 0.0,
            "count": 0
        }
    }
    # Load existing summary if available to keep error history
    summary_path = os.path.join(result_dir, "_0summary.json")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(result_dir, "summary.json")
        
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                old_result = json.load(f)
                result["error_info"] = old_result.get("error_info", {})
        except Exception as e:
            print(e)
            pass

    # Scan history files for timing
    history_files = glob.glob(os.path.join(result_dir, "history_*.json"))
    for hf in history_files:
        try:
            with open(hf, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if "timing" in meta:
                    t = meta["timing"]
                    for k in result["timing_stats"]:
                        if k != "count" and k in t:
                            result["timing_stats"][k] += t[k]
                    result["timing_stats"]["count"] += 1
        except Exception as e:
            pass

    finished_tasks = set()
    # Scan actions folder to get current status counts
    actions_dir = os.path.join(result_dir, "actions")
    if os.path.exists(actions_dir):
        for f in glob.glob(os.path.join(actions_dir, "*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                score = data.get("score", -0.1)
                task_id = os.path.basename(f).split(".")[0]
                if score == 1.0:
                    result["correct"] += 1
                    finished_tasks.add(task_id)
                elif score >= 0.5:
                    result["partially_correct"] += 1
                    finished_tasks.add(task_id)
                elif score >= 0.0:
                    if "ERROR:" in data.get("final_answer", ""):
                        result["error"] += 1
                    else:
                        result["incorrect"] += 1
                    finished_tasks.add(task_id)
                else:
                    # -0.1 means unfinished/error during execution but file created
                    pass
            except Exception as e:
                print(e)
                pass

    # Calculate error count: tasks in error_info that are NOT finished
    for job_json in result["error_info"]:
        tid = os.path.basename(job_json).split(".")[0]
        if tid not in finished_tasks:
            result["error"] += 1

    result["total"] = (
        result["correct"]
        + result["incorrect"]
        + result["partially_correct"]
        + result["error"]
    )
    return result
