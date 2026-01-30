import os
import json

def analyze(result_dir="results"):
    if not os.path.exists(result_dir):
        print(f"Result directory '{result_dir}' not found. Skipping analysis using {os.getcwd()}.")
        return

    results = {}
    for d in os.listdir(result_dir):
        if d.startswith("."):
            continue
        dir_path = os.path.join(result_dir,d)
        if not os.path.isdir(dir_path):
            continue
        site = d.split("_")[0]
        if "_Qwen2.5-VL-32B-Instruct_"in d:
            model = "Qwen2.5-VL-32B-Instruct"
        elif "_Qwen2.5-VL-32B-Instruct-SFT_"in d:
            model = "Qwen2.5-VL-32B-Instruct-SFT"
        elif "Qwen2-VL-2B-Instruct" in d:
             model = "Qwen2-VL-2B-Instruct"
        else:
            # raise ValueError("model") 
            # Relaxed check to avoid crashing on new models
            parts = d.split('_')
            if len(parts) >= 2:
                 # heuristic: use generic model name or directory name
                 model = d
            else:
                 continue

        if not site in results:
            results[site] = {}
        if not model in results[site]:
            results[site][model] = {"total":0,"correct":0,"correct_rate":0,"correct_under_30":0}
        action_dir = os.path.join(dir_path,"actions")
        if os.path.exists(action_dir):
            for f in os.listdir(action_dir):
                if not f.endswith(".json"):
                    continue
                try:
                    task_rest = json.load(open(os.path.join(action_dir,f),"r", encoding="utf-8"))
                    results[site][model]['total'] +=1
                    if task_rest.get('score', 0)>0:
                         results[site][model]['correct'] +=1
                    if results[site][model]['total'] > 0:
                        results[site][model]['correct_rate'] = results[site][model]['correct'] / results[site][model]['total']
                    if len(task_rest.get('actions', []))<30:
                        results[site][model]['correct_under_30'] +=1
                except Exception as e:
                    print(f"Error reading {f}: {e}")
    print(json.dumps(results,indent=4,sort_keys=True))

if __name__ == "__main__":
    analyze()