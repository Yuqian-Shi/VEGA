# 题目难度分析工具

用于分析SQL题目的难度，通过多个维度进行综合评分。

## 功能特性

### 多维度难度分析
1. **表复杂度** (权重: 15%)
   - 单表查询: 1分
   - 多表查询: 2-5分 (根据表数量)

2. **关联关系复杂度** (权重: 15%)
   - 0个关联关系: 1分
   - 1-2个关联关系: 2分
   - 3-4个关联关系: 3分
   - 5-6个关联关系: 4分
   - 6个以上: 5分

3. **操作类型复杂度** (权重: 20%)
   - SELECT: 1分 (简单)
   - INSERT / UPDATE/DELETE: 5分 (困难)

4. **结果复杂度** (权重: 20%)
   - 0个结果: 1分
   - 1个结果: 1.5分
   - 2-3个结果: 2.5分
   - 4-5个结果: 3.5分
   - 6-10个结果: 4分
   - 10个以上: 5分

5. **SQL语句复杂度** (权重: 30%)
   - JOIN数量、WHERE条件、子查询、聚合函数等

### 难度等级划分
- **简单**: ≤1.5分
- **较简单**: 1.5-2.5分
- **中等**: 2.5-3.5分
- **较困难**: 3.5-4.5分
- **困难**: >4.5分

## 文件说明

### 主要脚本
- `question_difficulty_analyzer.py` - 难度分析器
- `quick_difficulty_check.py` - 快速分析单个题目
- `difficulty_config.yaml` - 配置文件
- `requirements.txt` - 依赖包列表

### 输入数据格式
支持JSONL格式，每行一个JSON对象，包含以下字段：
```json
{
  "question": "题目描述",
  "template_type": "single_table|multi_table",
  "operation_type": "SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER",
  "primary_table": "主表名",
  "related_tables": ["相关表1", "相关表2"],
  "used_relationships": [...],
  "answer": ["结果1", "结果2"],
  "sql_execute_result": [["结果1"], ["结果2"]],
  "sql": "SQL语句"
}
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 分析整个文件
```bash
# 使用配置文件
python question_difficulty_analyzer.py input.jsonl -c difficulty_config.yaml

# 指定输出文件
python question_difficulty_analyzer.py input.jsonl -o output.json -c difficulty_config.yaml

# 只输出汇总报告
python question_difficulty_analyzer.py input.jsonl --summary-only -c difficulty_config.yaml
```

### 2. 快速分析单个题目
```bash
# 使用默认配置
python quick_difficulty_check.py

# 使用自定义配置
python quick_difficulty_check.py -c difficulty_config.yaml
```

## 输出结果

### 详细分析结果
```json
{
  "question_id": "题目描述...",
  "overall_difficulty": {
    "score": 2.85,
    "level": "中等"
  },
  "dimension_scores": {
    "table_complexity": 2.0,
    "relationship_complexity": 3.0,
    "operation_complexity": 1.0,
    "result_complexity": 2.5,
    "sql_complexity": 2.0
  },
  "dimension_details": {...},
  "weights": {...},
  "config_source": "custom"
}
```

### 汇总报告
```json
{
  "total_questions": 100,
  "difficulty_distribution": {
    "简单": 20,
    "较简单": 30,
    "中等": 25,
    "较困难": 15,
    "困难": 10
  },
  "dimension_averages": {...},
  "overall_stats": {...},
  "config_used": {
    "weights": {...},
    "difficulty_levels": [...]
  }
}
```