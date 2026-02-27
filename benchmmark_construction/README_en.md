# EntWorld Benchmark Generation Framework

## Project Overview

A framework specifically designed for generating verifiable questions and GUI Agent tasks for self-hosted websites. Through database and multi-table correlation analysis, it progressively generates workflow abstractions, verifiable questions, and GUI Agent/Model evaluation tasks.

### Core Features

- **Database Schema Analysis**: Automatically analyzes the database structure and table relationships of target websites.
- **Workflow Discovery**: Generates workflows with business characteristics based on business logic.
- **Question Generation**: Automatically generates verifiable evaluation questions based on workflows.
- **Task Construction**: Creates GUI Agent evaluation tasks compatible with the WebArena format.

## Architecture

<!-- 详见 飞书链接：https://c7py2lortg.feishu.cn/wiki/IbaNw2my8iHqNEkbhFYcN1ymnub?from=from_copylink   密码：#492X678 -->
![Overview](/asserts/dataset_construction_overview.png)

## Supported Website Platforms (Example Platforms)

### Field Description

- **Default Password**: The initial login password for the Web interface, used for first login and system initialization.
- **Evaluation Password**: The password used by the agent to access the website during evaluation task execution. If consistent with the default password, no modification is needed; if inconsistent, the password must be changed to this value during initialization.
- **Database Account**: The username used to connect to the database, used for accessing the database during the task generation phase for schema analysis and data validation.
- **Database Password**: The password used with the database account to connect to the database during the task generation phase.

### Platform Information 

| Website Platform        | Business Type   | Database Type      | Web Port | DB Port | Frontend Account  | Default Password      | Evaluation Password      | Database Account       | Database Password             | Image Core Component Versions                                                                                                                                                                                                                              
|-------------|--------|------------|-------|------|-------|-----------|-----------|-------------|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| EspoCRM     | CRM System  | MySQL      | 9900  | 3900 | admin | password  | password       | espocrm     | database_password | espocrm: sha256:b095fd0b016ce2ed07374c9625f9c6dbb67f5535def417fc6208f27be8860c40;<br> mariadb: sha256:300929c28ab758f3322f12273e9e8b0f2233d8af06050bd1b9e17133cc5beb1a;                                                                 
| ZenTao      | Project Mgmt   | MySQL      | 9901  | 3901 | admin | Admin1234 | Admin1234      | zentao      | zentao            | mysql:8.0; <br> easysoft/zentao: sha256:4c0d2bc36dbee2433e9a01348298fd4dff3c15f5afb3a5cf83dd1f1d1fa99071;                                                                                                                               
| OpenProject | Project Mgmt   | PostgreSQL | 9902  | 3902 | admin | admin     | admin123456    | openproject | openproject       | postgres:13; <br> openproject/community:13；                                                                                                                                                                                             
| Veops CMDB  | CMDB   | MySQL      | 9903  | 3903 | demo  | 123456    | 123456         | cmdb        | 123456            | registry.cn-hangzhou.aliyuncs.com/veops/cmdb-db:2.5；<br> registry.cn-hangzhou.aliyuncs.com/veops/cmdb-cache:2.5；<br> registry.cn-hangzhou.aliyuncs.com/veops/cmdb-api:2.5.3；<br> registry.cn-hangzhou.aliyuncs.com/veops/cmdb-ui:2.5.3; 
| iTOP        | ITIL Mgmt | MySQL      | 9904  | 3904 | admin | admin1234 | admin1234      | itop        | It0pDbP@ss!       | elestio/mysql:8.0; elestio/itop: sha256:6a4bdf38597dfdef465af01141d776174f22c934cab1eff5de4dc45a03bba96a;                                                                                                                               
| Snipe-IT    | Asset Mgmt   | MySQL      | 9907  | 3907 | admin | password  | password       | snipeit     | changeme1234      | snipe/snipe-it: sha256:adb1ab73bd3417b55fd8bca6a2909170c80e4310a9237963ae22f46b5ab6d1c2;                                                                                                                                                

## Website Deployment and Initialization

Before using the framework for evaluation task generation, you need to deploy and initialize the target websites. We provide a detailed deployment guide containing Docker Compose deployment steps and data initialization procedures for all supported websites.

**Detailed Documentation**: Please refer to the [Website Deployment and Initialization Guide](sites/bootstrap.md)

This document includes：

- Docker Compose deployment steps for each website.
- Detailed data initialization process (with screenshots).
- Access addresses and default account information.

> **Tips**: After deployment, ensure the website is accessible and data initialization is complete before proceeding with workflow discovery and question generation.

## Quick Start

### Environment Requirements

- **Python**: 3.10+

### 1. Environment Setup

```bash
# Clone the project
git clone <repository-url>
cd table2task

# (Optional) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy the environment configuration file
cp env.example .env

# Edit the configuration file
vim .env
```

#### Environment Variable Configuration Instructions

```bash
# OpenAI API Configuration
OPENAI_API_KEY=xxx
OPENAI_BASE_URL=xxx 

# Python Path Configuration
PYTHONPATH=/path/to/your/python/environment
```

### 3. Load Environment Variables

```bash
# Use the provided script to load environment variables
source prepare.sh
```

## Configuration Management

### Adding a New Target Website

To add a new target website, you need to create a corresponding configuration file in the directory `config/` .

#### Configuration File Structure

Create a `config/xxx.yaml` file (xxx must comply with Hydra naming conventions, refer to: `itop.yaml`).

## Core Workflow

### 1. Workflow Discovery

Analyzes the database schema and structure, combining business configurations to generate workflows with business characteristics.

#### Output Description

- **Output Path**: `workflow_output/` 
- **Main File**: `workflow_output/<config_name>.json`
- **Cache File**: `workflow_output/business_analysis/<config_name>_<sha1>.json`

#### erformance Optimization Suggestions

- **Table Count Limit**: It is recommended to keep the number of effective tables under 100.
- **Core Table Filtering**: You can manually specify the core table range using the core_tables parameter.
- **Caching Mechanism**: The system automatically skips tables without data and caches analysis results.

#### Caching Mechanism Details

- **Workflow Result Cache**:
    - Path: `workflow_output/<config_name>.json`
    - Content: Table structures, table relationships, workflow-to-table mappings.

- **Business Analysis Cache**:
    - Path: `workflow_output/business_analysis/<config_name>_<sha1>.json`
    - Content: Analysis results of table business characteristics and relationship analysis.
    - The SHA1 is generated based on the table scope; the cache is reused for the same table scope.

### 2. Question Generation

Generates verifiable evaluation questions based on the discovered workflows.

#### Parameter Description

- `workflow_config`: Absolute path to the workflow configuration file.
- `target_count`: Number of target questions.
- `template_count`: Number of templates.
- `task_type`: Type of task（query, cud, all）

#### Output Description

- **Output Path**: `questions_bank/`
- **File Format**: JSONL
- **Demo File**: `demo_questions_bank/<config_name>.jsonl`

### 3. Task Generation

Converts questions into GUI Agent evaluation tasks in WebArena format.

#### Output Description

- **Output Path**: `tasks_bank/` 
- **File Format**: JSON
- **Naming Convention**: `{task_id}.json`
- **Correspondence**: Tasks correspond one-to-one with questions.

### Unified CLI Evaluation Entry Point

The CLI is based on Click + Hydra, allowing you to quickly complete the entire process of "Workflow Discovery → Question Generation → Task Construction" via command-line parameters.
#### Basic Command Format

```bash
python3 cli.py \
  --conf=<configuration group name> \
  [--config="key1=value1 key2=value2"] \
  <subcommand> [subcommand arguments]
```

- `--conf`：Required. Corresponds to the configuration group  (e.g., `zentao`、`espocrm`) in `config/<conf>.yaml`.
- `--config`：Used to temporarily override Hydra parameters. Supports multiple `key=value` pairs separated by spaces.

#### Subcommands and Parameters Overview

| Level       | Parameter                    | Source                | Type/Value               | Required | Default   | Description                                     |
|----------|-----------------------|-------------------|---------------------|------|-------|----------------------------------------|
| public       | `--conf`              | CLI option            | String                 | Yes    | None     | Specifies the platform configuration group, must correspond to `config/<conf>.yaml`.     |
| public       | `--config`            | CLI option → Hydra override | `key=value ...`     | No    | Empty     | Passes key-value pairs to Hydra, overriding any configuration field.                 |
| discover | `--platform`          | CLI option            | String                 | No    | Config value | Specifies the runtime platform name, overrides `platform` in the config.             |
| discover | `--core-tables`       | CLI option            | Comma-separated string             | No    | Config value | Sets the core table set to speed up schema analysis.                   |
| discover | `--max-tables`        | CLI option            | Integer                  | No    | Config value | Limits the number of tables involved in the analysis.                            |
| generate | `--workflow-config`   | CLI option            | Path                  | No    | Config value | Points to the `workflow_output/*.json` file. If empty, uses the default from the config. |
| generate | `--target-count`      | CLI option            | Integer                  | No    | Config value | Number of target questions. Adjust based on business scale.                      |
| generate | `--template-count`    | CLI option            | Integer                  | No    | Config value | Number of templates, affects question coverage.                           |
| generate | `--task-type`         | CLI option            | `query`/`cud`/`all` | No    | Config value | Controls the question type (select/create-update-delete/mixed).                      |
| generate | `--platform`          | CLI option            | String                 | No    | Config value | Can separately override the platform name during the question generation phase.                       |
| factory  | `--raw-question-dir`  | CLI option            | Path                  | No    | Config value | Reads questions from a directory in batches. Lower priority than `--raw-question-file`.   |
| factory  | `--raw-question-file` | CLI option            | Path                  | No    | Config value | Specifies a single `*.jsonl` question file.                    |
| factory  | `--max-task-id`       | CLI option            | Integer                  | No    | Config value | The starting/upper-limit ID for generated tasks, used to avoid conflicts with existing tasks.              |

> Effective Priority: Explicit CLI options (including Hydra overrides via `--config`) > Corresponding value in the configuration file > Hard-coded default value in the code.

#### Recommended Evaluation Deployment Process

1. **Workflow Discovery**
   ```bash
   python3 cli.py --conf=zentao discover \
     --platform=zentao
   ```
    - To limit the number of tables processed, add `--max-tables=80`.

2. **Question Generation**
   ```bash 
   # Using explicitly defined CLI parameters
   python3 cli.py --conf=zentao generate \
     --workflow-config="/Users/sundapeng/Project/nlp/webrlvr/table2task/workflow_output/ZenTao.json" \
     --target-count=1 \
     --template-count=1 \
     --task-type=query
   ``` 

   Or

   ```bash
   # Using Hydra to override configuration keys
   python3 cli.py --conf=zentao --config="task_generation.workflow_config=/Users/sundapeng/Project/nlp/webrlvr/table2task/workflow_output/ZenTao.json task_generation.target_count=1 task_generation.template_count=1 task_generation.task_type=query" generate
   ```

3. **Task Construction**
   ```bash
   # Using explicitly defined CLI parameters
   python3 cli.py --conf=zentao factory \
     --raw-question-file="/Users/sundapeng/Project/nlp/webrlvr/table2task/questions_bank/ZenTao/20250908145252-generated_tasks_enhanced.jsonl" \
   ```

   Or

   ```bash
   # Using Hydra to override configuration keys
   python3 cli.py --conf=zentao --config="task_factory.raw_question_file=/Users/sundapeng/Project/nlp/webrlvr/table2task/questions_bank/ZenTao/20250908145252-generated_tasks_enhanced.jsonl" factory
   ```
    - If a directory `--raw-question-dir` is provided, all question files in that directory will be processed in batches.

## Question Difficulty Analysis Tool

Used to analyze the difficulty of SQL questions through scoring across multiple dimensions.

### Difficulty Level Classification

- **Easy**: ≤1.5 points
- **Fairly Easy**: 1.5-2.5 points
- **Medium**: 2.5-3.5分
- **Fairly Difficult**: 3.5-4.5 points
- **Difficult**: >4.5 points

See [analysis/README_en.md](benchmmark_construction/src/analysis/README_en.md) for Details.

## Database Support

### Supported Database Types

The framework currently supports the following database types:

- **MySQL**: Use the `mysql-connector-python` driver
- **PostgreSQL**: Use the `psycopg2` driver

### Extending to Other Database Types

To connect to other database types, extend the framework based on the architecture `helper/db_conn_factory.py`.

#### 1. Implement a DatabaseAdapter Subclass

Create a new adapter class that inherits from the `DatabaseAdapter` base class and implements all abstract methods.

#### 2. Update DatabaseConnectionFactory

Add the corresponding mapping in the `DatabaseConnectionFactory` class.

#### 3. Update the Configuration File

Specify the new database type in the configuration file:

```yaml
database:
  type: "your_database"  # Specify the new database type
  host: localhost
  port: 5432
  user: username
  password: password
  database: your_database
  # Other necessary configuration parameters
```

#### 4. Install Dependencies

Ensure the corresponding database driver dependency is added to `requirements.txt`:

```
your-database-driver==x.x.x
```