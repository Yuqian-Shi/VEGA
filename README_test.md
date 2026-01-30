# EntWorld: A Holistic Environment and Benchmark for Verifiable Enterprise GUI Agents

<!-- [<a href="https://jykoh.com/vwa">Website</a>]  -->
[<a href="https://arxiv.org/abs/xxx">Paper</a>]

<i>EntWorld</i> is a realistic and diverse benchmark about enterprise systems, which is for evaluating multimodal autonomous language agents. It comprises of a set of diverse tasks across 6 core business applications. It ensures reproducibility and executable evaluation. We propose a rigorous evaluation metric based on SQL state verification during dataset construction. By directly querying the underlying databases of the applications, EntWorld enables precise validation of task completion (e.g., verifying exact database record insertions or updates), ensuring deterministic and noise-free evaluation. This eliminates ambiguities in visual matching and enables high-precision correctness assessment.

![Overview](figures/dataset_construction_overview.png)

Here is the scores on test set results of EntWorld. All metrics are task Success Rate (SR). 
![Mainresults](figures/main_results.png)
<!-- ## TODOs -->
<!-- - [x] Add human trajectories.
- [x] Add GPT-4V + SoM trajectories from our paper. -->
<!-- - [x] Add scripts for end-to-end training and reset of environments. -->
<!-- - [x] Add demo to run multimodal agents on any arbitrary webpage. -->


## EntWorld Benchmmark Construction
if you want to know about the construction of EntWorld Benchmmark, you can following the instructions [here](https://xxx) for details.

## Install
```bash
# Python 3.10 (or 3.11, but not 3.12 cause 3.12 deprecated distutils needed here)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install
pip install -e .
```

You can also run the unit tests to ensure that VisualWebArena is installed correctly:
```
pytest -x
```

## End-to-end Evaluation
1. Setup the standalone environments.
Please check out [this page](environment_docker/README.md) for details.

2. Configurate the urls for each website.

```bash
export ESPOCRM='http://${SERVER}:9900'
export ZENTAO='http://${SERVER}:9901'
export OPENPROJECT='http://${SERVER}:9902'
export VEOPS_CMDB='http://${SERVER}:9903'
export ITOP='http://${SERVER}:9904'
export SNIPEIT='http://${SERVER}:9907'
```

3. Generate config files for each website test example:
Each website test example have a [config file] (./config_files) in this environment, which is used to reset, interacttion, and evaluating. Before the formal evaluation, it is also necessary to verify the website login configuration, The operations are as follows.

Generate config files:
```bash
python scripts/generate_test_data.py
```

Obtain and save the auto-login cookies for all websites:
```
bash prepare.sh
```
The configurations in this work are implemented based on [WebArena](https://github.com/web-arena-x/webarena). You can refer to [this link](https://github.com/web-arena-x/webarena) for more. You aslo can check [VisualAgentBench](https://github.com/THUDM/VisualAgentBench/tree/main/VAB-WebArena-Lite) to get the configurations.


4. Launch the evaluation. For example, to reproduce our GPT-4.1 captioning baseline:
After the configurations of the test examples, you can run the agents for evaluations. In this evaluation, the trajectory will be saved in `<your_result_dir>/0.html` if want. you can run the evalution with the following script. If you can chagnge the model for other baseline agent. 
```bash
python run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_3s.json \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --result_dir <your_result_dir> \
  --test_config_base_dir=config_files/ops/snipeit \
  --model gpt-4.1 \
  --mode chat \
  --provider openai \
  --max_obs_length 8192 \
  --max_tokens 2048 \
  --viewport_width 1280 \
  --viewport_height 2480 \
  --action_set_type id_accessibility_tree \
  --observation_type accessibility_tree
```

### Agent Trajectories

We analyzed the data results and provided examples of both successes and failures. The following is a trajectory of enterprise system web task, which display the evaluation process. The agent's observations and output at each step are shown.
![Demo](figures/snipit_83_rl.pdf)


## Citation
If you find our environment or our models useful, please consider citing our work Entworld:
```
@article{ying2026entworld,
  title={EntWorld: A Holistic Environment and Benchmark for Verifiable Enterprise GUI Agents},
  author={xx},
  journal={x},
  year={2026}
}
```

# VEGA
