import os
import json
import glob
from collections import defaultdict
import logging

def generate_report(base_result_dir="results"):
    print("\n" + "="*80)
    print(f"Generating Report from: {base_result_dir}")
    print("="*80 + "\n")
    
    # Find all summary.json files in subdirectories
    summary_files = glob.glob(os.path.join(base_result_dir, "*", "_0summary.json")) 
    
    if not summary_files:
        print("No result summary files found.")
        return

    site_stats = defaultdict(lambda: {
        "correct": 0, "partially_correct": 0, "incorrect": 0, "error": 0, "total": 0
    })
    
    for summary_file in summary_files:
        # parent folder name: site_model_hash
        dir_name = os.path.basename(os.path.dirname(summary_file))
        # This parsing is brittle if site name has underscores.
        # Better to read site name from config.json inside the folder
        config_path = os.path.join(os.path.dirname(summary_file), "config.json")
        site_name = dir_name # fallback
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                    site_name = cfg.get("site", {}).get("site", dir_name)
            except:
                pass
        
        try:
            with open(summary_file, "r") as f:
                data = json.load(f)
                
            site_stats[site_name]["correct"] += data.get("correct", 0)
            site_stats[site_name]["partially_correct"] += data.get("partially_correct", 0)
            site_stats[site_name]["incorrect"] += data.get("incorrect", 0)
            site_stats[site_name]["error"] += data.get("error", 0)
            site_stats[site_name]["total"] += data.get("total", 0)
            
        except Exception as e:
            print(f"Error reading {summary_file}: {e}")

    # Generate Markdown Table
    headers = ["Site", "Correct", "Partial", "Incorrect", "Error", "Total", "Success Rate (Corr+Part)"]
    rows = []
    
    total_stats = {k.lower(): 0 for k in headers if k not in ["Site", "Success Rate (Corr+Part)"]}
    
    for site, stats in sorted(site_stats.items()):
        success_rate = 0
        if stats["total"] > 0:
            success_rate = (stats["correct"] + stats["partially_correct"]) / stats["total"] * 100
        
        row = [
            site,
            stats["correct"],
            stats["partially_correct"],
            stats["incorrect"],
            stats["error"],
            stats["total"],
            f"{success_rate:.1f}%"
        ]
        rows.append(row)
        
        total_stats["correct"] += stats["correct"]
        total_stats["partial"] += stats["partially_correct"]
        total_stats["incorrect"] += stats["incorrect"]
        total_stats["error"] += stats["error"]
        total_stats["total"] += stats["total"]

    # Total Row
    total_success_rate = 0
    if total_stats["total"] > 0:
        total_success_rate = (total_stats["correct"] + total_stats["partial"]) / total_stats["total"] * 100
    
    rows.append([
        "**TOTAL**",
        f"**{total_stats['correct']}**",
        f"**{total_stats['partial']}**",
        f"**{total_stats['incorrect']}**",
        f"**{total_stats['error']}**",
        f"**{total_stats['total']}**",
        f"**{total_success_rate:.1f}%**"
    ])

    # Print Table
    col_widths = [max(len(str(val)) for val in col) for col in zip(*([headers] + rows))]
    # Add some padding
    col_widths = [w + 2 for w in col_widths]
    
    header_row = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    
    # Separator row
    # Just standard dashed line
    print(header_row)
    print("-" * len(header_row))
    
    for row in rows:
        print("".join(f"{str(val):<{w}}" for val, w in zip(row, col_widths)))

    print("\n" + "="*80 + "\n")
