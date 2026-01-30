import argparse
import json
import re
from typing import Any, Optional
import numpy as np

import tiktoken
from beartype import beartype
from PIL import Image
import time
import os
from vega.agent.prompts import (
    PromptConstructor,
    CoTPromptConstructor,
    MultimodalCoTPromptConstructor,
    DirectPromptConstructor,
    WebRLPromptConstructor,
    WebRLChatPromptConstructor,
)
from vega.browser_env import Trajectory
from vega.browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_webrl_id_based_action,
)
from vega.browser_env.utils import Observation, StateInfo
from vega.llms import (
    call_llm,
    generate_from_huggingface_completion,
    lm_config,
)

# Optional imports for OpenAI
try:
    from vega.llms import (
        generate_from_openai_chat_completion,
        generate_from_openai_completion,
    )
except ImportError:
    generate_from_openai_chat_completion = None
    generate_from_openai_completion = None

from vega.llms.tokenizers import Tokenizer


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        captioning_fn=None,
        planner_ip=None,
        result_dir: str = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn
        self.planner_ip = planner_ip
        self.result_dir = result_dir
        # Use multimodal_inputs setting from lm_config, which comes from all_config.yaml
        self.multimodal_inputs = (
            lm_config.multimodal_inputs
            and type(prompt_constructor) == MultimodalCoTPromptConstructor
        )

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def next_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        images: Optional[list[Image.Image]] = None,
        output_response: bool = False,
    ) -> Action:
        # Create page screenshot image for multimodal models.
        if self.multimodal_inputs:
            obs = trajectory[-1]["observation"]
            page_screenshot_arr = obs["image"]
            page_screenshot_img = Image.fromarray(page_screenshot_arr)  
            # size = (viewport_width, viewport_width)

        # Caption the input image, if provided.
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"
            elif not self.multimodal_inputs:
                print("WARNING: Input image provided but no image caption available.")

        if self.multimodal_inputs:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, page_screenshot_img, images, meta_data
            )
        else:
            prompt = self.prompt_constructor.construct(trajectory, intent, meta_data)
        lm_config = self.lm_config
        # input(lm_config)
        n = 0
        while True:
            if self.planner_ip is not None and self.planner_ip != "":
                response = call_llm(lm_config, prompt, None, base_url=self.planner_ip)
            else:
                response = call_llm(lm_config, prompt)
            force_prefix = self.prompt_constructor.instruction["meta_data"].get(
                "force_prefix", ""
            )
            response = f"{force_prefix}{response}"

            history = meta_data.get("history")
            if not history:
                meta_data["history"] = [(prompt[-1], response)]
            else:
                meta_data["history"].append((prompt[-1], response))

            if output_response:
                print(f"Agent: {response}", flush=True)
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(response)
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "webrl_id":
                    action = create_webrl_id_based_action(parsed_response)
                else:
                    raise ValueError(f"Unknown action type {self.action_set_tag}")
                action["raw_prediction"] = response
                break
            except ActionParsingError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break

        return action

    def reset(self, test_config_file: str) -> None:
        pass


def construct_agent(task_cfg, captioning_fn=None, result_dir=None) -> Agent:
    agent: Agent
    model_cfg = task_cfg["model"]
    agent_type = model_cfg["agent_type"]
    
    if agent_type == "prompt":
        instruction_meta = task_cfg["model"].get("instruct_meta", {})
        constructor_type = model_cfg.get("prompt_constructor", instruction_meta.get("prompt_constructor"))
        
        if not model_cfg.get("multimodal_inputs", False) and constructor_type == "MultimodalCoTPromptConstructor":
            constructor_type = "CoTPromptConstructor"
        
        lm_cfg = lm_config.construct_llm_config(model_cfg)
        
        # Mapping constructors instead of using eval() for better security and clarity
        constructors = {
            "CoTPromptConstructor": CoTPromptConstructor,
            "MultimodalCoTPromptConstructor": MultimodalCoTPromptConstructor,
            "DirectPromptConstructor": DirectPromptConstructor,
            "WebRLPromptConstructor": WebRLPromptConstructor,
            "WebRLChatPromptConstructor": WebRLChatPromptConstructor,
        }
        
        constructor_class = constructors.get(constructor_type)
        if not constructor_class:
            raise ValueError(f"Unknown prompt constructor type: {constructor_type}")

        prompt_constructor = constructor_class(
            model_cfg["instruction_path"],
            lm_config=lm_cfg,
            tokenizer=Tokenizer(model_cfg["provider"], model_cfg["model"]),
        )
        
        agent = PromptAgent(
            action_set_tag=model_cfg.get("action_type", instruction_meta.get("action_type")),
            lm_config=lm_cfg,
            prompt_constructor=prompt_constructor,
            planner_ip=model_cfg.get("planner_ip", None),
            captioning_fn=captioning_fn,
            result_dir=result_dir,
        )
        return agent
    else:
        raise NotImplementedError(f"agent type {agent_type} not implemented")
