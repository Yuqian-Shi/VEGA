#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速难度检查脚本
用于快速分析单个题目的难度
"""

import json
import argparse
from question_difficulty_analyzer import QuestionDifficultyAnalyzer


def quick_analyze_question(question_data, config_file=None):
    """快速分析单个题目"""
    analyzer = QuestionDifficultyAnalyzer(config_file=config_file)
    result = analyzer.analyze_question(question_data)
    
    print("=== 题目难度分析结果 ===")
    print(f"题目: {result['question_id']}")
    print(f"总体难度: {result['overall_difficulty']['level']} ({result['overall_difficulty']['score']:.2f})")
    
    print("\n各维度得分:")
    for dimension, score in result['dimension_scores'].items():
        print(f"  {dimension}: {score:.2f}")
    
    print("\n详细分析:")
    
    # 表复杂度详情
    table_details = result['dimension_details']['table_complexity']
    print(f"表复杂度: {table_details['template_type']}, 总表数: {table_details['total_tables']}")
    
    # 关联关系详情
    rel_details = result['dimension_details']['relationship_complexity']
    print(f"关联关系: {rel_details['relationship_count']} 个")
    
    # 操作类型详情
    op_details = result['dimension_details']['operation_complexity']
    print(f"操作类型: {op_details['operation_type']}")
    
    # 结果复杂度详情
    res_details = result['dimension_details']['result_complexity']
    print(f"结果数量: {res_details['result_count']}")
    
    # SQL复杂度详情
    sql_details = result['dimension_details']['sql_complexity']
    print(f"SQL复杂度: JOIN数={sql_details['join_count']}, 条件数={sql_details['where_conditions']}")
    
    # 配置信息
    print(f"\n配置来源: {result.get('config_source', 'unknown')}")
    print(f"权重配置: {result['weights']}")
    
    return result


def main():
    """主函数 - 示例用法"""
    parser = argparse.ArgumentParser(description='快速难度检查')
    parser.add_argument('-c', '--config', help='配置文件路径 (YAML格式)')
    args = parser.parse_args()
    
    # 示例题目数据
    sample_question = {
        "question": "As a service manager, I need to count services with status production to evaluate current operational assets.",
        "template_type": "single_table",
        "operation_type": "SELECT",
        "primary_table": "service",
        "related_tables": [],
        "used_relationships": [],
        "answer": ["3"],
        "sql_execute_result": [["3"]],
        "sql": "SELECT COUNT(*) FROM `service` t1 WHERE t1.`status` = 'production'"
    }
    
    print("分析示例题目...")
    result = quick_analyze_question(sample_question, config_file=args.config)
    
    print("\n" + "="*50)
    print("JSON格式结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
