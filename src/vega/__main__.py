import multiprocessing
from vega.common.helpers import load_configuration
from vega.runner.pipeline import run_evaluation_pipeline

def main():
    multiprocessing.freeze_support()
    
    config_manager, logger = load_configuration()
    config_manager.print_summary()
    
    run_evaluation_pipeline(config_manager, logger)
    
    logger.info("Evaluation pipeline completed.")

if __name__ == "__main__":
    main()
