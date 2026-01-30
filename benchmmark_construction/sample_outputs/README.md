# 目录用途
该目录为输出示例，帮助理解生成的结构与字段含义。

# 文件结构
- questions_bank/: 问题样例（jsonl）
- tasks_bank/: 任务样例（json）

# 数据格式示例
- questions_bank/*.jsonl: 每行一个对象
- tasks_bank/*.json: 任务定义，对应 webarena 类的任务

# 复现
1) 配置 env 与 config.yaml
2) 运行 generate_questions.py / task_factory.py
3) 产出将与本示例结构一致