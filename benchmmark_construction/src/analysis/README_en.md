# Question Difficulty Analysis Tool

Used to analyze the difficulty of SQL questions through scoring across multiple dimensions.

## Features

### Difficulty Analysis
1. **Table Complexity** (Weight: 15%)
   - Single-table query: 1 points
   - Multi-table query: 2-5 points(based on number of tables)

2. **Relationship Complexity** (Weight: 15%)
   - 0 relationships: 1 points
   - 1-2 relationships: 2 points
   - 3-4 relationships: 3 points
   - 5-6 relationships: 4 points
   - More than 6: 5 points

3. **Operation Type Complexity** (Weight: 20%)
   - SELECT: 1 point (Easy)
   - INSERT / UPDATE/DELETE: 5 points (Difficult)

4. **Result Complexity** (Weight: 20%)
   - 0 results: 1 point
   - 1 result: 1.5 points
   - 2-3 results: 2.5 points
   - 4-5 results: 3.5 points
   - 6-10 results: 4 points
   - More than 10: 5 points

5. **SQL Statement Complexity** (Weight: 30%)
   - Number of JOINs, WHERE conditions, subqueries, aggregate functions, etc.

### Difficulty Level Classification
- **Easy**: ≤1.5 points
- **Fairly Easy**: 1.5-2.5 points
- **Medium**: 2.5-3.5分
- **Fairly Difficult**: 3.5-4.5 points
- **Difficult**: >4.5 points

## File Description

### Main Scripts
- `question_difficulty_analyzer.py` - Difficulty analyzer
- `quick_difficulty_check.py` - Quick analysis of single question
- `difficulty_config.yaml` - Configuration file
- `requirements.txt` - List of dependencies

### Input Data Format
Supports JSONL format, each line containing a JSON object with the following fields:
```json
{
  "question": "Question description",
  "template_type": "single_table|multi_table",
  "operation_type": "SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER",
  "primary_table": "Primary table name",
  "related_tables": ["related_table1", "related_table2"],
  "used_relationships": [...],
  "answer": ["result1", "result2"],
  "sql_execute_result": [["result1"], ["result2"]],
  "sql": "SQL statement"
}
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Use

### 1. Analyze an Entire File
```bash
# Using configuration file
python question_difficulty_analyzer.py input.jsonl -c difficulty_config.yaml

# Specify output file
python question_difficulty_analyzer.py input.jsonl -o output.json -c difficulty_config.yaml

# Output summary report only
python question_difficulty_analyzer.py input.jsonl --summary-only -c difficulty_config.yaml
```

### 2. Quick Analysis of a Single Question
```bash
# Using default configuration
python quick_difficulty_check.py

# Using custom configuration
python quick_difficulty_check.py -c difficulty_config.yaml
```

## Output Results

### Detailed Analysis Results
```json
{
  "question_id": "Question description...",
  "overall_difficulty": {
    "score": 2.85,
    "level": "Medium"
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

### Summary Report
```json
{
  "total_questions": 100,
  "difficulty_distribution": {
    "Easy": 20,
    "Fairly Easy": 30,
    "Medium": 25,
    "Fairly Difficult": 15,
    "Difficult": 10
  },
  "dimension_averages": {...},
  "overall_stats": {...},
  "config_used": {
    "weights": {...},
    "difficulty_levels": [...]
  }
}
```