import os
import glob
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from vega.runner.site import run_site_wrapper
from vega.runner.models import ready_model
from vega.browser_env.env_config import reload_config
from vega.analysis import analyze

def run_evaluation_pipeline(config_manager, logger):
    """Main execution pipeline for the VEGA benchmark."""
    
    reload_config(config_manager.config_data)

    active_sites = config_manager.active_sites
    active_models = config_manager.active_models
    
    # Calculate total tasks
    total_tasks = 0
    for site_name, site_config in active_sites.items():
        task_dir = site_config.get("task_dir")
        st_idx = site_config.get("start_idx", 0)
        ed_idx = site_config.get("end_idx")
        if ed_idx is None or ed_idx < st_idx:
             task_files = glob.glob(os.path.join(task_dir, "[0-9]*.json"))
             ed_idx = len(task_files)
        
        site_tasks_count = ed_idx - st_idx
        sample_rate = site_config.get("sample_rate")
        if sample_rate is None:
            sample_rate = config_manager.config_data.get("sample_rate", 1.0)
            
        if sample_rate < 1.0:
             site_tasks_count = max(1, int(site_tasks_count * sample_rate))

        total_tasks += site_tasks_count
    
    total_tasks *= len(active_models)
    
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    logger.info(f"Starting evaluations with {total_tasks} total tasks...")
    
    max_workers = config_manager.config_data.get("cli", {}).get("max_workers", 0)
    if max_workers == 0:
        max_workers = os.cpu_count() or 1
        logger.info(f"max_workers set to 0, using CPU count: {max_workers}")

    with tqdm(total=total_tasks, desc="Processing Tasks") as pbar:
        for model_name, model_config in active_models.items():
            logger.info(f"Processing model: {model_name}")
            
            try:
                ready_model(model_config, logger)
            except Exception as e:
                logger.error(f"Failed to prepare model {model_name}: {e}")
                raise e
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for site_name, site_config in active_sites.items():
                    futures.append(executor.submit(
                        run_site_wrapper, 
                        site_config, 
                        model_config, 
                        config_manager.config_data, 
                        config_manager.log_path, 
                        progress_queue
                    ))
                
                completed = 0
                while completed < total_tasks:
                    try:
                        progress_queue.get(timeout=1)
                        pbar.update(1)
                        completed += 1
                    except:
                        if all(f.done() for f in futures):
                            break
    
    logger.info("Evaluation pipeline completed.")
    
    # Run analysis
    result_dir = config_manager.config_data.get("output", {}).get("result_dir")
    if result_dir and os.path.exists(result_dir):
        logger.info(f"Analyzing results in {result_dir}...")
        analyze(result_dir)

