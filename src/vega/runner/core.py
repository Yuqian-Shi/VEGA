import time
import os
import json
import logging
import traceback
import sys
import subprocess
import tempfile
from pathlib import Path

from vega.browser_env import (
    ActionTypes,
    create_stop_action,
    Trajectory, 
    StateInfo,
    ScriptBrowserEnv
)
from vega.browser_env.helper_functions import RenderHelper
from vega.browser_env.auto_login import get_site_comb_from_filepath
from vega.evaluation_harness import evaluator_router, llm_fuzzy_match
from vega.agent import construct_agent

from .utils import (
    save_action_history,
    check_early_stop,
    execute_action_step,
    load_task_images,
    setup_task_directories,
    save_webrl_traces,
    ConfigContainer
)

def perform_auto_login(job_cfg, task_cfg, job_json_path, logger):
    if not job_cfg["storage_state"]:
        return job_json_path

    cookie_file_name = os.path.basename(job_cfg["storage_state"])
    comb = get_site_comb_from_filepath(cookie_file_name)
    temp_dir = tempfile.mkdtemp()

    sites_config = {}
    for site_name in comb:
        if task_cfg["site"]["site"].startswith(site_name):
            sites_config[site_name] = task_cfg["site"]

    # Locate auto_login.py relative to the browser_env package or current file
    # Assuming runner/core.py, going up one level to root, then browser_env/auto_login.py
    root_dir = os.path.dirname(os.path.dirname(__file__))
    login_script_path = os.path.join(
        root_dir, "browser_env", "auto_login.py"
    )
    if not os.path.exists(login_script_path):
         # Fallback try relative to this file if structure is different
         login_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "browser_env", "auto_login.py"))

    logger.debug(f"logging in with: {comb} {sites_config}")
    result = subprocess.run(
        [
            sys.executable,
            login_script_path,
            "--auth_folder",
            temp_dir,
            "--site_list",
            *comb,
            "--sites_config",
            json.dumps(sites_config),
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"Auto login failed with return code {result.returncode}")
        logger.error(f"Stdout: {result.stdout}")
        logger.error(f"Stderr: {result.stderr}")
        raise RuntimeError(f"Auto login failed: {result.stderr}")

    job_cfg["storage_state"] = os.path.join(temp_dir, cookie_file_name)
    assert os.path.exists(
        job_cfg["storage_state"]
    ), f"Storage state file not created: {job_cfg['storage_state']}"

    new_job_cfg_path = os.path.join(temp_dir, os.path.basename(job_json_path))
    with open(new_job_cfg_path, "w", encoding="utf-8") as f:
        json.dump(job_cfg, f)

    return new_job_cfg_path

def evaluate_task_result(
    job_cfg_path,
    trajectory,
    task_cfg,
    sbe,
    logger,
    render_helper,
    task_id,
    actions,
    out_path,
    result_dir,
    meta_data=None,
    captioning_fn=None,
):
    evaluator = evaluator_router(job_cfg_path, captioning_fn=captioning_fn)

    correctness_check = task_cfg["model"].get("correctness_check", task_cfg.get("correctness_check"))
    if (
        len(trajectory) > 0
        and isinstance(trajectory[-1], dict)
        and "answer" not in trajectory[-1]
    ):
        trajectory[-1]["answer"] = ""

    start_verify = time.time()
    
    last_action = trajectory[-1]
    final_answer = last_action.get("answer", "")
    if "Early stop" in final_answer or "ERROR" in final_answer:

         if "余额不足" in final_answer:
             logger.critical(f"[Result] (FATAL) {job_cfg_path} - Insufficient Balance: {final_answer}. Quitting.")
             sys.exit(0)

         retry_indicators = [
             "content management policy",
             "该令牌额度已用尽", 
             "Connection refused",
             "Connection timed out",
             "Network is unreachable",
             "500 Internal Server Error",
             "502 Bad Gateway",
             "503 Service Unavailable",
             "504 Gateway Time-out",
             "Rate limit", 
             "429",
             "APIConnectionError",
             "ServiceUnavailable",
             "RemoteDisconnected",
             "ProtocolError"
         ]
         
         is_retryable = any(ind in final_answer for ind in retry_indicators)

         if is_retryable:
             score = -0.1 
             ret_status = "error"
             logger.info(f"[Result] (ERROR) {job_cfg_path} - Retryable Error: {final_answer}")
             save_action_history(out_path, task_id, actions=actions, score=score, final_answer=final_answer)
             render_helper.close()
             return ret_status, final_answer
         

    score = evaluator(
        trajectory=trajectory,
        config_file=job_cfg_path,
        page=sbe.page,
        model_cfg=correctness_check,
        result_dir=result_dir,
        task_id=task_id,
    )
    
    if correctness_check and score < 1.0:
        try:
            last_action = trajectory[-1]
            pred = last_action.get("answer", "")
            
            if pred:
                with open(job_cfg_path, "r", encoding="utf-8") as f:
                    job_cfg_data = json.load(f)
                
                intent = job_cfg_data["intent"]
                ref_answers = job_cfg_data.get("eval", {}).get("reference_answers", {})
                
                refs = []
                if isinstance(ref_answers, dict):
                    for k, v in ref_answers.items():
                        if k in ["exact_match", "must_include", "fuzzy_match"]:
                            if isinstance(v, str):
                                refs.append(v)
                            elif isinstance(v, list):
                                refs.extend(v)
                
                if refs:
                    reference = " |OR| ".join(refs)
                    logger.info(f"Running explicit LLM verification for task {task_id}")
                    verify_score = llm_fuzzy_match(
                        pred=pred,
                        reference=reference,
                        question=intent,
                        model_cfg=correctness_check,
                        result_dir=result_dir,
                        task_id=task_id
                    )
                    if verify_score > score:
                        score = verify_score
                        logger.info(f"LLM verification updated score to {score}")
        except Exception as e:
            logger.warning(f"Failed to run explicit LLM verification: {e}")

    if meta_data and "timing" in meta_data:
        meta_data["timing"]["verify_model"] = time.time() - start_verify

    final_answer = trajectory[-1].get("answer", "")
    save_action_history(out_path, task_id, actions=actions, score=score, final_answer=final_answer)

    if score == 1:
        logger.info(f"[Result] (PASS) {job_cfg_path}")
        ret = "correct"
    elif score >= 0.5:
        logger.info(f"[Result] (PARTIAL) {job_cfg_path}")
        ret = "partially_correct"
    else:
        if final_answer and "ERROR:" in final_answer:
            logger.info(f"[Result] (ERROR) {job_cfg_path}")
            ret = "error"
        else:
            logger.info(f"[Result] (FAIL) {job_cfg_path}")
            ret = "incorrect"

    if task_cfg["browser"]["save_trace_enabled"]:
        traces_dir = task_cfg.get("output", {}).get("traces_dir", "traces")
        sbe.save_trace(
            Path(task_cfg["site"]["result_dir"]) / traces_dir / f"{task_id}.zip"
        )

    site_name = task_cfg["site"]["site"]
    model_name = task_cfg["model"]["model"]
    task_num = os.path.basename(job_cfg_path).split(".")[0]
    logger.info(f"Task Completed: ({site_name}, {model_name}, {task_num})")

    render_helper.close()
    return ret, trajectory[-1].get("answer", "")

def initialize_benchmark_task(task_cfg, agent, env, job_json, logger):
    render_helper = RenderHelper(
        job_json,
        task_cfg["site"]["result_dir"],
        task_cfg["model"].get("action_type", task_cfg["model"].get("instruct_meta", {}).get("action_type", "id_accessibility_tree")),
    )
    with open(job_json, encoding="utf-8") as f:
        job_cfg = json.load(f)
    intent = job_cfg["intent"]
    task_id = job_cfg["task_id"]
    job_cfg["start_url"] = task_cfg["site"].get("url")

    job_cfg_path = perform_auto_login(job_cfg, task_cfg, job_json, logger)

    images = load_task_images(job_cfg.get("image_path", None))

    logger.info(f"[Config file]: {job_cfg_path}")
    try:
        logger.info(f"[Intent]: {intent}")
    except UnicodeEncodeError:
        safe_intent = intent.encode('ascii', 'replace').decode('ascii')
        logger.info(f"[Intent]: {safe_intent}")

    agent.reset(job_cfg_path)
    trajectory: Trajectory = []
    obs, info = env.reset(options={"config_file": job_cfg_path})
    state_info: StateInfo = {"observation": obs, "info": info}
    trajectory.append(state_info)
    meta_data = {"action_history": ["None"], "history": []}

    result_dir, out_path = setup_task_directories(task_cfg, task_id)

    actions = []
    max_steps = task_cfg["model"]["max_steps"]
    early_stop_thresholds = {
        "parsing_failure": task_cfg["model"]["early_stopping"][
            "parsing_failure_threshold"
        ],
        "repeating_action": task_cfg["model"]["early_stopping"][
            "repeating_action_threshold"
        ],
    }
    return (
        render_helper,
        intent,
        task_id,
        images,
        job_cfg_path,
        trajectory,
        state_info,
        meta_data,
        out_path,
        actions,
        max_steps,
        early_stop_thresholds,
        result_dir,
    )

def run_benchmark_task(task_cfg, agent, sbe, job_json, logger, result_dir=None, captioning_fn=None):
    (
        render_helper,
        intent,
        task_id,
        images,
        job_cfg_path,
        trajectory,
        state_info,
        meta_data,
        out_path,
        actions,
        max_steps,
        early_stop_thresholds,
        result_dir,
    ) = initialize_benchmark_task(task_cfg, agent, sbe, job_json, logger)
    
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                cached_res = json.load(f)
            
            if cached_res.get("score", -0.1) != -0.1:
                logger.info(f"Skipping task {task_id}: Found cached Stage 2 result (score={cached_res['score']})")
                render_helper.close()
                score = cached_res["score"]
                if score == 1: return "correct", "", {}
                elif score >= 0.5: return "partially_correct", "", {}
                else: return "incorrect", "", {}
            
            if "final_answer" in cached_res:
                logger.info(f"Using cached Stage 1 result for task {task_id}")
                actions = cached_res["actions"]
                final_answer = cached_res["final_answer"]
                trajectory = [state_info]
                trajectory.append({"action_type": ActionTypes.STOP, "answer": final_answer})
                
                ret, ans = evaluate_task_result(
                    job_cfg_path, trajectory, task_cfg, sbe, logger, render_helper, 
                    task_id, actions, out_path, result_dir, meta_data, captioning_fn=captioning_fn
                )
                return ret, ans, {}
        except Exception as e:
            logger.warning(f"Failed to read cache for task {task_id}: {e}")
            logger.warning(traceback.format_exc())
            trajectory = [state_info]

    meta_data["timing"] = {
        "llm_thinking": 0.0,
        "verify_model": 0.0,
        "browser_action": 0.0,
        "other": 0.0,
        "total_wall_time": 0.0
    }
    start_wall_time = time.time()

    enable_screenshot = task_cfg["browser"]["render_screenshot"]
    while True:
        save_action_history(out_path, task_id, actions=actions)
        early_stop_flag, stop_info = check_early_stop(
            trajectory, max_steps, early_stop_thresholds, actions
        )
        if early_stop_flag:
            action = create_stop_action(f"Early stop: {stop_info}")
        else:
            try:
                output_response = task_cfg["model"]["output_response"]
                
                start_llm = time.time()
                action = agent.next_action(
                    trajectory,
                    intent,
                    meta_data=meta_data,
                    output_response=output_response,
                    images=images,
                )
                
                meta_data["timing"]["llm_thinking"] += time.time() - start_llm
                
            except ValueError as e:
                logger.exception(e)
                logger.error(f"ValueError during action prediction: {e}")
                action = create_stop_action(f"ERROR: {str(e)}")
                print(e)
            except Exception as e:
                logger.exception(e)
                logger.error(f"Exception during action prediction: {e}")
                action = create_stop_action(f"ERROR: {str(e)}")
                print(e)

        trajectory.append(action)
        
        start_browser = time.time()
        execute_action_step(
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
        )
        cost = time.time() - start_browser
        meta_data["timing"]["browser_action"] += cost

        if action["action_type"] == ActionTypes.STOP:
            break
        try:
            start_browser_step = time.time()
            obs, _, terminated, _, info = sbe.step(action)
            meta_data["timing"]["browser_action"] += time.time() - start_browser_step
        except Exception as e:
            logger.warning(f"Action execution failed: {e}")
            obs = f"Error executing action: {str(e)}"
            terminated = False
            info = state_info.get("info", {})
            print(e)
        state_info = {"observation": obs, "info": info}
        trajectory.append(state_info)

        if terminated:
            trajectory.append(create_stop_action(""))
            break
        
        current_answer = None
        if trajectory and isinstance(trajectory[-1], dict) and "answer" in trajectory[-1]:
             current_answer = trajectory[-1]["answer"]
        save_action_history(out_path, task_id, actions=actions, final_answer=current_answer)

    traces_dir = task_cfg.get("output", {}).get("traces_dir", "traces")
    save_webrl_traces(sbe, trajectory, result_dir, task_id, intent, traces_dir)

    ret, ans = evaluate_task_result(
        job_cfg_path,
        trajectory,
        task_cfg,
        sbe,
        logger,
        render_helper,
        task_id,
        actions,
        out_path,
        result_dir=result_dir,
        meta_data=meta_data,
        captioning_fn=captioning_fn,
    )

    meta_data["timing"]["total_wall_time"] = time.time() - start_wall_time
    meta_data["timing"]["other"] = meta_data["timing"]["total_wall_time"] - (
        meta_data["timing"]["llm_thinking"] + 
        meta_data["timing"]["verify_model"] + 
        meta_data["timing"]["browser_action"]
    )
    
    logger.info(f"Timing Stats for Task {task_cfg['site']['site']}_{task_id}:")
    logger.info(f"  LLM Thinking:   {meta_data['timing']['llm_thinking']:.2f}s")
    logger.info(f"  Verify Model:   {meta_data['timing']['verify_model']:.2f}s")
    logger.info(f"  Browser Action: {meta_data['timing']['browser_action']:.2f}s")
    logger.info(f"  Other Overhead: {meta_data['timing']['other']:.2f}s")
    logger.info(f"  Total Wall Time:{meta_data['timing']['total_wall_time']:.2f}s")

    history_path = os.path.join(result_dir, f"history_{task_id}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4)

    return ret, ans, meta_data["timing"]

def evaluate_single_task(
    task_cfg, job_json, sbe, agent, logger, result_dir=None, captioning_fn=None
):
    ret = "error"
    return run_benchmark_task(
        task_cfg, agent, sbe, job_json, logger, result_dir=result_dir, captioning_fn=captioning_fn
    )
