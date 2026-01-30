"""
Unified configuration loader for VEGA (Verifiable Enterprise GUI Agents)

This module provides a single entry point for all configuration needs,
consolidating the long config chain into one YAML file.

Usage:
    config = ConfigManager("all_config.yaml")
    
    # Access configurations
    browser_config = config.get_browser_config()
    llm_config = config.get_llm_config()
"""

import os
import argparse
import json
import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict

import yaml


@dataclass
class BrowserConfig:
    """Browser environment configuration."""
    render_screenshot: bool
    slow_mo: int
    viewport_width: int
    viewport_height: int
    current_viewport_only: bool
    save_trace_enabled: bool
    sleep_after_execution: float
    
    @property
    def headless(self) -> bool:
        """Headless mode is the opposite of render_screenshot."""
        return not self.render_screenshot


@dataclass
class AgentConfig:
    """Agent configuration."""
    agent_type: str
    instruction_path: str
    parsing_failure_threshold: int
    repeating_action_threshold: int
    max_steps: int
    planner_ip: str
    output_response: bool


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    test_start_idx: int
    test_end_idx: int
    task_dir: str


@dataclass
class OutputConfig:
    """Output and logging configuration."""
    result_dir: str
    log_files_dir: str
    traces_dir: str
    error_file: str
    config_file: str


class ConfigManager:
    """Unified configuration manager for the entire application."""
    
    def __init__(self, config_file):
        """Initialize config manager.
        
        Args:
            config_file: Path to the YAML configuration file
        """
        self.config_file = Path(config_file)
        self.config_data = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            self.config_data = yaml.safe_load(f)
        
        # Get global SSH profile if available
        ssh_profile_config = self.config_data.get("ssh_profile") or {}
        llm_server_profile = ssh_profile_config.get("llm_server")
        self.active_sites = {}
        self.active_models = {}
        for model in self.config_data.get("llm").get("active_models"):
            self.active_models[model] = self.config_data["llm"]["model_with_config"].get(model)
            
            # Inject global SSH profile if not present in model config
            if llm_server_profile and "ssh_profile" not in self.active_models[model]:
                self.active_models[model]['ssh_profile'] = llm_server_profile

            instruction_path = self.active_models[model].get("instruction_path")
            self.active_models[model]['model'] = model
            self.active_models[model]['start_cmd'] = self.active_models[model]['start_cmd'].replace("MODEL_ID", model)
            with open(instruction_path, 'r') as instr_file:
                self.active_models[model]['instruct_meta'] = json.load(instr_file).get("meta_data", {})
        for site in self.config_data.get("sites").get("active_sites"):
            self.active_sites[site] = self.config_data["sites"]["sites_with_config"].get(site)
            self.active_sites[site]['site'] = site
    def get_browser_config(self) -> BrowserConfig:
        """Get browser configuration."""
        if "browser" not in self.config_data:
            raise KeyError("Missing top-level 'browser' configuration section")
        browser = self.config_data["browser"]
        try:
            viewport = browser["viewport"]
            return BrowserConfig(
                render_screenshot=browser["render_screenshot"],
                slow_mo=browser["slow_mo"],
                viewport_width=viewport["width"],
                viewport_height=viewport["height"],
                current_viewport_only=browser["current_viewport_only"],
                save_trace_enabled=browser["save_trace_enabled"],
                sleep_after_execution=browser["sleep_after_execution"],
            )
        except KeyError as e:
            raise KeyError(f"Missing browser configuration key: {e}")
    
    def get_agent_config(self) -> AgentConfig:
        """Get agent configuration (strict, per-model from LLM config).
        
        Agent configuration is now primarily per-model in llm.model_with_config.
        This method returns default/placeholder values for backward compatibility.
        """
        return AgentConfig(
            agent_type=None,
            instruction_path=None,
            parsing_failure_threshold=None,
            repeating_action_threshold=None,
            max_steps=None,
            planner_ip=None,
            output_response=None,
        )

    
        correctness = self.config_data["correctness_check"]
    def get_dataset_config(self) -> DatasetConfig:
        """Get dataset configuration from active sites.
        
        Retrieves configuration from the first active site's config.
        """
        sites = self.config_data["sites"]
        active_sites = sites["active_sites"]
        if not active_sites:
            raise ValueError("No active sites specified in sites.active_sites")
        
        active_site = active_sites[0]
        sites_with_config = sites["sites_with_config"]
        if active_site not in sites_with_config:
            raise KeyError(f"Active site '{active_site}' not found in sites.sites_with_config")
        
        site_cfg = sites_with_config[active_site]
        return DatasetConfig(
            test_start_idx=site_cfg["start_idx"],
            test_end_idx=site_cfg["end_idx"],
            task_dir=site_cfg["task_dir"],
        )

    def get_output_config(self) -> OutputConfig:
        """Get output/logging configuration from active sites.
        
        Retrieves configuration from the first active site's config.
        """
        sites = self.config_data["sites"]
        active_sites = sites["active_sites"]
        if not active_sites:
            raise ValueError("No active sites specified in sites.active_sites")
        
        active_site = active_sites[0]
        sites_with_config = sites["sites_with_config"]
        if active_site not in sites_with_config:
            raise KeyError(f"Active site '{active_site}' not found in sites.sites_with_config")
        
        site_cfg = sites_with_config[active_site]
        return OutputConfig(
            result_dir=site_cfg["result_dir"],
            log_files_dir=site_cfg["log_files_dir"],
            traces_dir=site_cfg["traces_dir"],
            error_file=site_cfg["error_file"],
            config_file=site_cfg["config_file"],
        )
    
        
        return args
    
    def save(self, output_file: Optional[str] = None) -> None:
        """Save current configuration to YAML file.
        
        Args:
            output_file: Path to save to (defaults to original file)
        """
        output_path = Path(output_file or self.config_file)
        with open(output_path, 'w') as f:
            yaml.dump(self.config_data, f, default_flow_style=False, sort_keys=False)
        print(f"Config saved to {output_path}")
    
    def print_summary(self) -> None:
        """Print a summary of current configuration."""
        browser = self.get_browser_config()
        output = self.get_output_config()
        correctness = self.config_data["correctness_check"]
        
        # Get site-specific task ranges
        sites_config = self.config_data.get("sites")
        active_sites = sites_config.get("active_sites")
        sites_with_config = sites_config.get("sites_with_config")
        
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        print("Active Models")
        for model in self.config_data.get("llm").get("active_models"):
            print(f"  - {model}")
            model_config = self.config_data.get("llm", {}).get("model_with_config", {}).get(model, {})
            print(json.dumps(model_config, indent=4))
            
        print("Active Sites")
        for site in active_sites:
            print(f"  - {site}:")
            # print(json.dumps(sites_with_config.get(site), indent=4))
        print("\nBrowser Configuration:")
        print(json.dumps(asdict(browser), indent=4))

        print("\nOutput Configuration:")
        print(json.dumps(asdict(output), indent=4))
        print("\nCorrectness Check Configuration:")
        print(json.dumps(correctness, indent=4))
        print("="*70 + "\n")
        print("End of Configuration Summary\n")


if __name__ == "__main__":
    # Example usage
    config = ConfigManager("all_config.yaml")
    
    config.print_summary()
    
    # Get specific configs
    print("Browser config:", config.get_browser_config())
    
    # Convert to argparse.Namespace for existing code
