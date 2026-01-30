#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Authors :       sundapeng.sdp
   Date：          2025/9/1
   Description :   题目难度分析器, 多路策略分析题目难度，包括表数量、关联关系、操作类型、结果复杂度等
-------------------------------------------------
"""
__author__ = 'sundapeng.sdp'

import json
import argparse
from typing import Dict, List, Any, Tuple
from pathlib import Path
import re
import yaml


class QuestionDifficultyAnalyzer:
    """题目难度分析器"""
    
    def __init__(self, config_file: str = None):
        # 加载配置
        self.config = self.load_config(config_file)
        
        # 从配置中获取设置
        self.weights = self.config['weights']
        self.operation_difficulty = self.config['operation_difficulty']
        self.table_complexity_rules = self.config['table_complexity_rules']
        self.relationship_complexity_rules = self.config['relationship_complexity_rules']
        self.result_complexity_rules = self.config['result_complexity_rules']
        self.sql_complexity_rules = self.config['sql_complexity_rules']
        self.difficulty_levels = self.config['difficulty_levels']
    
    def load_config(self, config_file: str = None) -> Dict[str, Any]:
        """加载配置文件"""
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                print(f"已加载配置文件: {config}")
                
            except Exception as e:
                print(f"加载配置文件失败: {e}, 使用默认配置")
        else:
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        return config
    
    def analyze_table_complexity(self, question_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """分析表复杂度"""
        template_type = question_data.get('template_type', 'single_table')
        primary_table = question_data.get('primary_table', '')
        related_tables = question_data.get('related_tables', [])
        
        total_tables = 1 + len(related_tables)
        
        # 使用配置规则计算分数
        if template_type == 'single_table':
            table_score = self.table_complexity_rules['single_table']
        elif template_type == 'multi_table':
            table_score = self.table_complexity_rules['multi_table'].get(
                total_tables, 
                self.table_complexity_rules['multi_table']['default']
            )
        else:
            table_score = 0.0
        
        details = {
            'template_type': template_type,
            'total_tables': total_tables,
            'primary_table': primary_table,
            'related_tables': related_tables,
            'score': table_score
        }
        
        return table_score, details
    
    def analyze_relationship_complexity(self, question_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """分析关联关系复杂度"""
        used_relationships = question_data.get('used_relationships', [])
        
        # 使用配置规则计算分数
        relationship_count = len(used_relationships)
        relationship_score = 1.0
        
        for rule in self.relationship_complexity_rules['count_ranges']:
            if relationship_count <= rule['max']:
                relationship_score = rule['score']
                break
        
        # 分析关联关系类型
        relationship_types = {}
        for rel in used_relationships:
            rel_type = rel.get('relationship_type', 'unknown')
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        details = {
            'relationship_count': relationship_count,
            'relationship_types': relationship_types,
            'score': relationship_score
        }
        
        return relationship_score, details
    
    def analyze_operation_complexity(self, question_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """分析操作类型复杂度"""
        operation_type = question_data.get('operation_type', 'SELECT')
        
        operation_score = self.operation_difficulty.get(operation_type, 0.0)
        
        normalized_score = min(2.0, max(0.0, operation_score))
        
        details = {
            'operation_type': operation_type,
            'base_score': operation_score,
            'normalized_score': normalized_score
        }
        
        return normalized_score, details
    
    def analyze_result_complexity(self, question_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """分析结果复杂度"""
        answer = question_data.get('answer', [])
        sql_execute_result = question_data.get('sql_execute_result', [])
        
        result_count = len(answer) if answer else 0
        
        # 使用配置规则计算分数
        result_score = 1.0
        for rule in self.result_complexity_rules['count_ranges']:
            if result_count <= rule['max']:
                result_score = rule['score']
                break
        
        # 分析结果数据类型复杂度
        data_complexity = 1.0
        if sql_execute_result:
            for row in sql_execute_result:
                if isinstance(row, list) and len(row) > 1:
                    data_complexity += 0.5
                if any(isinstance(item, (int, float)) for item in row if isinstance(row, list)):
                    data_complexity += 0.3
        
        final_score = min(5, (result_score + data_complexity) / 2)
        
        details = {
            'result_count': result_count,
            'data_complexity': data_complexity,
            'result_score': result_score,
            'final_score': final_score
        }
        
        return final_score, details
    
    def analyze_sql_complexity(self, question_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """分析SQL语句复杂度"""
        sql = question_data.get('sql', '')
        
        # 使用配置规则计算分数
        rules = self.sql_complexity_rules
        complexity_score = rules['base_score']
        
        join_count = sql.upper().count('JOIN')
        if join_count > 0:
            complexity_score += min(2.0, join_count * rules['join_multiplier'])
        
        where_conditions = len(re.findall(r'WHERE|AND|OR', sql.upper()))
        if where_conditions > 0:
            complexity_score += min(1.5, where_conditions * rules['where_multiplier'])
        
        if re.search(r'\(.*SELECT.*\)', sql, re.IGNORECASE):
            complexity_score += rules['subquery_bonus']
        
        if re.search(r'COUNT|SUM|AVG|MAX|MIN|GROUP BY', sql.upper()):
            complexity_score += rules['aggregation_bonus']
        
        final_score = min(rules['max_score'], max(1.0, complexity_score))
        
        details = {
            'join_count': join_count,
            'where_conditions': where_conditions,
            'has_subquery': bool(re.search(r'\(.*SELECT.*\)', sql, re.IGNORECASE)),
            'has_aggregation': bool(re.search(r'COUNT|SUM|AVG|MAX|MIN|GROUP BY', sql.upper())),
            'complexity_score': complexity_score,
            'final_score': final_score
        }
        
        return final_score, details
    
    def calculate_overall_difficulty(self, scores: Dict[str, float]) -> Tuple[float, str]:
        """计算总体难度分数和等级"""
        weighted_sum = sum(scores[key] * self.weights[key] for key in self.weights.keys())
        
        # 使用配置的难度等级
        level = "困难"  # 默认值
        for level_rule in self.difficulty_levels:
            if weighted_sum <= level_rule['max_score']:
                level = level_rule['level']
                break
        
        return weighted_sum, level
    
    def analyze_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析单个题目的难度"""
        # 分析各个维度
        table_score, table_details = self.analyze_table_complexity(question_data)
        relationship_score, relationship_details = self.analyze_relationship_complexity(question_data)
        operation_score, operation_details = self.analyze_operation_complexity(question_data)
        result_score, result_details = self.analyze_result_complexity(question_data)
        sql_score, sql_details = self.analyze_sql_complexity(question_data)
        
        # 汇总各维度分数
        scores = {
            'table_complexity': table_score,
            'relationship_complexity': relationship_score,
            'operation_complexity': operation_score,
            'result_complexity': result_score,
            'sql_complexity': sql_score
        }
        
        # 计算总体难度
        overall_score, difficulty_level = self.calculate_overall_difficulty(scores)
        
        # 构建分析结果
        analysis_result = {
            'question_id': question_data.get('question', '')[:50] + '...' if len(question_data.get('question', '')) > 50 else question_data.get('question', ''),
            'overall_difficulty': {
                'score': round(overall_score, 2),
                'level': difficulty_level
            },
            'dimension_scores': {
                'table_complexity': round(table_score, 2),
                'relationship_complexity': round(relationship_score, 2),
                'operation_complexity': round(operation_score, 2),
                'result_complexity': round(result_score, 2),
                'sql_complexity': round(sql_score, 2)
            },
            'dimension_details': {
                'table_complexity': table_details,
                'relationship_complexity': relationship_details,
                'operation_complexity': operation_details,
                'result_complexity': result_details,
                'sql_complexity': sql_details
            },
            'weights': self.weights,
            'config_source': 'custom' if hasattr(self, '_config_loaded') else 'default'
        }
        
        return analysis_result
    
    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """分析文件中的所有题目"""
        results = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    question_data = json.loads(line)
                    analysis_result = self.analyze_question(question_data)
                    analysis_result['line_number'] = line_num
                    results.append(analysis_result)
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行JSON解析失败: {e}")
                    continue
        
        return results
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成汇总报告"""
        if not results:
            return {}
        
        # 统计难度分布
        difficulty_distribution = {}
        for result in results:
            level = result['overall_difficulty']['level']
            difficulty_distribution[level] = difficulty_distribution.get(level, 0) + 1
        
        # 计算各维度平均分
        dimension_averages = {}
        for dimension in ['table_complexity', 'relationship_complexity', 'operation_complexity', 'result_complexity',
                          'sql_complexity']:
            scores = [result['dimension_scores'][dimension] for result in results]
            dimension_averages[dimension] = {
                'average': round(sum(scores) / len(scores), 2),
                'min': round(min(scores), 2),
                'max': round(max(scores), 2)
            }
        
        # 总体难度统计
        overall_scores = [result['overall_difficulty']['score'] for result in results]
        overall_stats = {
            'average': round(sum(overall_scores) / len(overall_scores), 2),
            'min': round(min(overall_scores), 2),
            'max': round(max(overall_scores), 2)
        }
        
        summary = {
            'total_questions': len(results),
            'difficulty_distribution': difficulty_distribution,
            'dimension_averages': dimension_averages,
            'overall_stats': overall_stats,
            'analysis_timestamp': str(Path().absolute()),
            'config_used': {
                'weights': self.weights,
                'difficulty_levels': self.difficulty_levels
            }
        }
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='题目难度分析')
    parser.add_argument('input_file', help='输入的JSONL文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('--summary-only', action='store_true', help='只输出汇总报告')
    parser.add_argument('-c', '--config', default="difficulty_config.yaml", help='配置文件路径')
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"输入文件 {args.input_file} 不存在")
        return
    
    # 创建分析器，支持配置文件
    analyzer = QuestionDifficultyAnalyzer(config_file=args.config)
    
    print(f"正在分析文件: {args.input_file}")
    
    results = analyzer.analyze_file(args.input_file)
    
    if not results:
        print("没有找到有效的题目数据")
        return
    
    print(f"成功分析 {len(results)} 个题目")
    
    summary = analyzer.generate_summary_report(results)
    
    if args.summary_only:
        output_data = summary
    else:
        output_data = {
            'summary': summary,
            'detailed_results': results
        }
    
    if args.output:
        output_path = args.output
    else:
        input_stem = Path(args.input_file).stem
        output_path = f"{input_stem}_difficulty_analysis.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"分析结果已保存到: {output_path}")
    
    print("\n=== 难度分析汇总 ===")
    print(f"总题目数: {summary['total_questions']}")
    print(f"总体难度范围: {summary['overall_stats']['min']} - {summary['overall_stats']['max']}")
    print(f"总体难度平均: {summary['overall_stats']['average']}")
    
    print("\n难度分布:")
    for level, count in summary['difficulty_distribution'].items():
        percentage = (count / summary['total_questions']) * 100
        print(f"  {level}: {count} 题 ({percentage:.1f}%)")
    
    print("\n各维度平均分:")
    for dimension, stats in summary['dimension_averages'].items():
        print(f"  {dimension}: {stats['average']} (范围: {stats['min']} - {stats['max']})")
    
    print(f"\n配置信息:")
    print(f"  权重配置: {analyzer.weights}")
    print(f"  难度等级: {[level['level'] for level in analyzer.difficulty_levels]}")


if __name__ == "__main__":
    main()
