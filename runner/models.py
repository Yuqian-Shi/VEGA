import os
import subprocess
import time
import logging
import requests
import hashlib
import json
from pathlib import Path
from utils.consts import MODEL_ID_KEY, MODEL_BASE_URL_KEY

def generate_result_dir(task_config, base_result_dir="results"):
    gene_llm_cfg = task_config
    gene_llm_name = gene_llm_cfg["model"]["model"]
    site_name = task_config["site"]["site"]
    
    model_config = task_config["model"]
    gen_related_keys = [
        "instruction_path",
        "agent_type",
        "multimodal_inputs",
        "generation",
        "early_stopping",
        "max_steps",
        "captioning_model",
        "model",
        "provider",
        "mode"
    ]
    
    filtered_model_config = {
        k: model_config[k] for k in gen_related_keys if k in model_config
    }
    
    if "generation" in filtered_model_config and isinstance(filtered_model_config["generation"], dict):
        gen_dict = filtered_model_config["generation"].copy()
        for key in ["base_url", "api_key", "model_endpoint"]:
            if key in gen_dict:
                del gen_dict[key]
        filtered_model_config["generation"] = gen_dict

    verify_config = task_config.get("correctness_check", {})
    verify_related_keys = ["model", "temperature", "top_p", "max_tokens", "provider"]
    filtered_verify_config = {
        k: verify_config[k] for k in verify_related_keys if k in verify_config
    }

    hash_payload = {
        "site": site_name,
        "generate_llm": filtered_model_config,
        "verify_llm": filtered_verify_config,
    }
    
    hash_val = hashlib.md5(
        json.dumps(hash_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:6]

    result_dir = os.path.join(
        base_result_dir, f"{site_name}_{gene_llm_name}_{hash_val}"
    )
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    return result_dir


def get_ssh_prefix(profile):
    if not profile:
        return []
    ssh_options = ["-o", "StrictHostKeyChecking=no", "-o", f"UserKnownHostsFile={'NUL' if os.name == 'nt' else '/dev/null'}"]
    return ["ssh"] + ssh_options + [profile]

def ensure_no_vllm(model_config, logger):
    ssh_profile = model_config.get("ssh_profile")
    
    cmd_prefix = get_ssh_prefix(ssh_profile)
    
    # Prepare check and kill commands based on environment
    if ssh_profile:
        # Remote Linux assumed
        check_cmd = ["ps -ef | grep vllm.entrypoints.openai.api_server | grep -v grep"]
        kill_cmd = ["for pid in $(ps -ef | grep vllm.entrypoints.openai.api_server | grep -v grep | awk '{print $2}'); do kill -15 $pid; done"]
    else:
        # Local
        if os.name == 'nt':
            # Windows Local
            check_cmd = ['powershell', '-Command', "Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -match 'vllm.entrypoints.openai.api_server'}"]
            kill_cmd = ['powershell', '-Command', "Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -match 'vllm.entrypoints.openai.api_server'} | Invoke-CimMethod -MethodName Terminate"]
        else:
            # Linux Local
            check_cmd = ["bash", "-c", "ps -ef | grep vllm.entrypoints.openai.api_server | grep -v grep"]
            kill_cmd = ["bash", "-c", "for pid in $(ps -ef | grep vllm.entrypoints.openai.api_server | grep -v grep | awk '{print $2}'); do kill -15 $pid; done"]

    logger.info("Ensuring no vLLM processes are running...")
    
    max_retries = 3
    for i in range(max_retries):
        # Check if running
        if ssh_profile:
             # For SSH, append the command string
             full_check_cmd = cmd_prefix + check_cmd
             full_kill_cmd = cmd_prefix + kill_cmd
        else:
            # Local
            full_check_cmd = check_cmd
            full_kill_cmd = kill_cmd

        check_res = subprocess.run(full_check_cmd, capture_output=True, text=True)
        
        is_running = False
        if ssh_profile or os.name != 'nt':
            if check_res.returncode == 0 and check_res.stdout.strip():
                is_running = True
        else:
            # Windows
            if check_res.stdout.strip():
                is_running = True

        if not is_running:
            logger.info("No vLLM process found.")
            return

        logger.info(f"Found vLLM process, sending Terminate (Attempt {i+1}/{max_retries})...")
        
        subprocess.run(full_kill_cmd, check=False)
        
        wait_timeout = 60
        start_wait = time.time()
        while time.time() - start_wait < wait_timeout:
            check_res = subprocess.run(full_check_cmd, capture_output=True, text=True)
            
            still_running = False
            if ssh_profile or os.name != 'nt':
                 if check_res.returncode == 0 and check_res.stdout.strip():
                     still_running = True
            else:
                 if check_res.stdout.strip():
                     still_running = True
            
            if not still_running:
                logger.info("vLLM process terminated.")
                return
            time.sleep(2)
    
    raise RuntimeError("Failed to kill vLLM process after multiple attempts")

def get_server_model(base_url, logger):
    response = requests.get(base_url+"/models/", timeout=5)
    return response.json()['data'][0]['id']

def wait_for_model(base_url,model_name, logger):
    health_url  = base_url
    logger.info(f"Waiting for model to be ready at {health_url}...")
    
    start_time = time.time()
    timeout = 60*30 
    health_url += "/models/"
    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            json_resp = response.json()
            if response.status_code == 200:
                if model_name in [m['id'] for m in json_resp.get('data', [])]:
                    logger.info(f"{model_name} is ready!")
                    return
        except Exception as e:
            pass
        time.sleep(5)
    
    raise TimeoutError(f"Model {model_name} failed to become ready at {health_url} within {timeout} seconds")

def ready_model(model_config, logger):
    target_model = model_config[MODEL_ID_KEY]
    try:
        cur_model = get_server_model(model_config['generation'][MODEL_BASE_URL_KEY], logger)
    except:
        cur_model = None
    if cur_model == target_model:
        return
    start_cmd = model_config.get("start_cmd")
    
    ensure_no_vllm(model_config, logger)

    ssh_profile = model_config.get("ssh_profile")
    cmd_prefix = get_ssh_prefix(ssh_profile)
    
    if ssh_profile:
        cmd = cmd_prefix + [start_cmd]
        logger.info(f"Executing SSH command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True,timeout=600)
    else:
        # Local execution
        logger.info(f"Executing Local command: {start_cmd}")
        subprocess.run(start_cmd, shell=True, check=True)

    wait_for_model(model_config['generation'][MODEL_BASE_URL_KEY], target_model, logger)
