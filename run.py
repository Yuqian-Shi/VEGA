# encoding: utf-8
import time
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import multiprocessing
import concurrent.futures
import glob
import collections
from tqdm import tqdm
from utils.utils import load_configuration
from runner.site import run_site_wrapper
from runner.models import ready_model
from reporting.aggregator import generate_report
from browser_env.env_config import reload_config
from analyze_results import analyze # Import the analysis function

def main():
    multiprocessing.freeze_support() # For Windows
    
    config_manager, logger = load_configuration()
    config_manager.print_summary()
    
    # Initialize URL_MAPPINGS globally with the loaded config
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
            
            model_tasks = []
            
            # Smart task distribution among workers
            num_sites = len(active_sites)
            if max_workers > num_sites and num_sites > 0:
                 # If we have more workers than sites, split sites into chunks
                 workers_per_site = max_workers // num_sites
                 extra_workers = max_workers % num_sites
            else:
                 workers_per_site = 1
                 extra_workers = 0

            site_keys = list(active_sites.keys())
            for idx, site_name in enumerate(site_keys):
                site_config = active_sites[site_name]
                
                # Determine workers for this site
                n_workers = workers_per_site
                if idx < extra_workers:
                    n_workers += 1

                if n_workers <= 1:
                     model_tasks.append((site_config, model_config, config_manager.config_data, config_manager.log_path, progress_queue))
                else:
                    # Logic to split the site into n_workers chunks
                    task_dir = site_config.get("task_dir")
                    st_base = site_config.get("start_idx", 0)
                    ed_base = site_config.get("end_idx")
                    
                    # Auto-detect end_idx if needed
                    if ed_base is None or ed_base < st_base:
                         task_files = glob.glob(os.path.join(task_dir, "[0-9]*.json"))
                         real_ed = len(task_files)
                    else:
                         real_ed = ed_base
                    
                    total_site_tasks = real_ed - st_base
                    if total_site_tasks <= 0:
                        # Should not happen typically, but fallback
                        model_tasks.append((site_config, model_config, config_manager.config_data, config_manager.log_path, progress_queue))
                        continue
                        
                    chunk_size = (total_site_tasks + n_workers - 1) // n_workers
                    
                    for i in range(n_workers):
                        chunk_st = st_base + i * chunk_size
                        chunk_ed = min(st_base + (i + 1) * chunk_size, real_ed)
                        
                        if chunk_st >= chunk_ed:
                            break
                            
                        # Create split config
                        split_site_config = site_config.copy()
                        split_site_config["start_idx"] = chunk_st
                        split_site_config["end_idx"] = chunk_ed
                        split_site_config["_is_split"] = True # Marker for logging if needed
                        
                        model_tasks.append((split_site_config, model_config, config_manager.config_data, config_manager.log_path, progress_queue))

            # Use ProcessPoolExecutor to run sites in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for task in model_tasks:
                    futures.append(executor.submit(run_site_wrapper, task))
                    time.sleep(1)
                
                finished_sites = 0
                while finished_sites < len(model_tasks):
                    while not progress_queue.empty():
                        try:
                            progress_queue.get_nowait()
                            pbar.update(1)
                        except Exception:
                            break
                    
                    done_count = sum(1 for f in futures if f.done())
                    if done_count > finished_sites:
                        finished_sites = done_count
                    
                    if finished_sites == len(model_tasks) and progress_queue.empty():
                        break
                    
                    time.sleep(0.1)
            
                for f in futures:
                    try:
                        f.result()
                    except Exception as e:
                        logger.error(f"Site execution failed: {e}")

    logger.info("All tasks completed.")
    
    # Generate aggregated report
    # Try to find user result dir from config
    output_cfg = config_manager.config_data.get("output", {})
    if isinstance(output_cfg, dict):
        result_dir = output_cfg.get("result_dir", "results")
    else:
        result_dir = "results"
        
    generate_report(result_dir)
    
    # Run detailed analysis
    try:
        print("\nRunning Analysis...")
        analyze(result_dir)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()
