# 基于数据库分析的 GUI Agent 评测任务生成框架

## 项目概述

专门针对 Self-host 网站进行可验证题目和 GUI Agent 评测任务集生成的框架。通过数据库和多表关联分析方法，逐步生成工作流抽象、可验证题目以及
GUI Agent/Model 评测任务。

### 核心功能

- **数据库 Schema 分析**: 自动分析目标网站的数据库结构和表关系
- **工作流发现**: 基于业务逻辑生成具有业务特性的工作流
- **题目生成**: 根据工作流自动生成可验证的评测题目
- **任务构建**: 生成符合 WebArena 格式的 GUI Agent 评测任务

## 架构

详见 飞书链接：https://c7py2lortg.feishu.cn/wiki/IbaNw2my8iHqNEkbhFYcN1ymnub?from=from_copylink   密码：#492X678

## 支持的网站平台（示例平台）

### 字段说明

- **默认密码**: Web 界面初始登录密码，用于首次登录和系统初始化
- **评测密码**: 评测任务执行时智能体访问网站使用的密码。如果与默认密码一致，则无需修改；如不一致，需在初始化时将密码修改为此值
- **数据库账户**: 连接数据库时使用的用户名，用于任务生成阶段访问数据库进行 Schema 分析和数据验证
- **数据库密码**: 连接数据库时使用的密码，与数据库账户配合使用，用于任务生成阶段访问数据库

### 平台信息表

| 网站平台        | 业务类型   | 数据库类型      | Web端口 | DB端口 | 前端账户  | 默认密码      | 评测密码      | 数据库账户       | 数据库密码             | 镜像核心组件版本                                                                                                                                                                                                                                
|-------------|--------|------------|-------|------|-------|-----------|-----------|-------------|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| EspoCRM     | CRM系统  | MySQL      | 9900  | 3900 | admin | password  | password       | espocrm     | database_password | espocrm: sha256:b095fd0b016ce2ed07374c9625f9c6dbb67f5535def417fc6208f27be8860c40;<br> mariadb: sha256:300929c28ab758f3322f12273e9e8b0f2233d8af06050bd1b9e17133cc5beb1a;                                                                 
| ZenTao      | 项目管理   | MySQL      | 9901  | 3901 | admin | Admin1234 | Admin1234      | zentao      | zentao            | mysql:8.0; <br> easysoft/zentao: sha256:4c0d2bc36dbee2433e9a01348298fd4dff3c15f5afb3a5cf83dd1f1d1fa99071;                                                                                                                               
| OpenProject | 项目管理   | PostgreSQL | 9902  | 3902 | admin | admin     | admin123456    | openproject | openproject       | postgres:13; <br> openproject/community:13；                                                                                                                                                                                             
| Veops CMDB  | CMDB   | MySQL      | 9903  | 3903 | demo  | 123456    | 123456         | cmdb        | 123456            | registry.cn-hangzhou.aliyuncs.com/veops/cmdb-db:2.5；<br> registry.cn-hangzhou.aliyuncs.com/veops/cmdb-cache:2.5；<br> registry.cn-hangzhou.aliyuncs.com/veops/cmdb-api:2.5.3；<br> registry.cn-hangzhou.aliyuncs.com/veops/cmdb-ui:2.5.3; 
| iTOP        | ITIL管理 | MySQL      | 9904  | 3904 | admin | admin1234 | admin1234      | itop        | It0pDbP@ss!       | elestio/mysql:8.0; elestio/itop: sha256:6a4bdf38597dfdef465af01141d776174f22c934cab1eff5de4dc45a03bba96a;                                                                                                                               
| Snipe-IT    | 资产管理   | MySQL      | 9907  | 3907 | admin | password  | password       | snipeit     | changeme1234      | snipe/snipe-it: sha256:adb1ab73bd3417b55fd8bca6a2909170c80e4310a9237963ae22f46b5ab6d1c2;                                                                                                                                                

## 网站部署与初始化

在使用框架进行评测任务生成之前，需要先部署和初始化目标网站。我们提供了详细的部署指南，包含所有支持网站的 Docker Compose
部署步骤和数据初始化流程。

**详细文档**: 请参考 [网站部署与初始化指南](sites/bootstrap.md)

该文档包含：

- 各网站的 Docker Compose 部署步骤
- 数据初始化详细流程（含截图说明）
- 访问地址和默认账户信息

> **提示**: 部署完成后，请确保网站可正常访问，并且已完成数据初始化，然后再进行后续的工作流发现和题目生成操作。

## Quick Start

### 环境要求

- **Python 版本**: 3.10+

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd table2task

# （可选）创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境配置

```bash
# 复制环境配置文件
cp env.example .env

# 编辑配置文件
vim .env
```

#### 环境变量配置说明

```bash
# OpenAI API 配置
OPENAI_API_KEY=xxx
OPENAI_BASE_URL=xxx 

# Python 路径配置
PYTHONPATH=/path/to/your/python/environment
```

### 3. 加载环境变量

```bash
# 使用提供的脚本加载环境变量
source prepare.sh
```

## 配置管理

### 新增目标网站

要添加新的目标网站，需要在 `config/` 目录下创建对应的配置文件。

#### 配置文件结构

创建 `config/xxx.yaml` 文件（xxx 需符合 Hydra 命名规则，参考：`itop.yaml`）

## 核心工作流程

### 1. 工作流发现 (Workflow Discovery)

分析数据库 Schema 和结构，结合业务配置生成具有业务特性的工作流。

#### 输出说明

- **输出路径**: `workflow_output/` 目录
- **主要文件**: `workflow_output/<config_name>.json`
- **缓存文件**: `workflow_output/business_analysis/<config_name>_<sha1>.json`

#### 性能优化建议

- **表数量限制**: 建议有效表数量控制在 100 张以内
- **核心表筛选**: 可通过 `core_tables` 参数手动指定核心表范围
- **缓存机制**: 系统自动跳过无数据表，并缓存分析结果

#### 缓存机制详解

- **Workflow 结果缓存**:
    - 路径: `workflow_output/<config_name>.json`
    - 内容: 表结构、表关系、工作流与表对应关系

- **业务分析缓存**:
    - 路径: `workflow_output/business_analysis/<config_name>_<sha1>.json`
    - 内容: 表业务特性和关联关系分析结果
    - SHA1 基于表范围生成，相同表范围复用缓存

### 2. 题目生成 (Question Generation)

基于工作流生成可验证的评测题目。

#### 参数说明

- `workflow_config`: 工作流配置文件绝对路径
- `target_count`: 目标题目数量
- `template_count`: 模板数量
- `task_type`: 任务类型（query, cud, all）

#### 输出说明

- **输出路径**: `questions_bank/` 目录
- **文件格式**: JSONL 格式
- **示例文件**: `demo_questions_bank/<config_name>.jsonl`

### 3. 任务构建 (Task Generation)

将题目转换为 WebArena 格式的 GUI Agent 评测任务。

#### 输出说明

- **输出路径**: `tasks_bank/` 目录
- **文件格式**: 每个任务一个 JSON 文件
- **命名规则**: `{task_id}.json`
- **对应关系**: 任务与题目一一对应

### CLI 统一评测入口

CLI 基于 Click + Hydra，支持通过命令行参数快速完成“工作流发现 → 题目生成 → 任务构建”全流程。

#### 基本命令格式

```bash
python3 cli.py \
  --conf=<配置组名称> \
  [--config="key1=value1 key2=value2"] \
  <子命令> [子命令参数]
```

- `--conf`：必填，对应 `config/<conf>.yaml` 中的配置组（例如 `zentao`、`espocrm`）。
- `--config`：用于临时覆盖 Hydra 参数，支持传入多个 `key=value`，以空格分隔。

#### 子命令与参数概览

| 层级       | 参数                    | 来源                | 类型/取值               | 是否必填 | 默认值   | 说明                                     |
|----------|-----------------------|-------------------|---------------------|------|-------|----------------------------------------|
| 公共       | `--conf`              | CLI 选项            | 字符串                 | 是    | 无     | 指定平台配置组，需与 `config/<conf>.yaml` 对应     |
| 公共       | `--config`            | CLI 选项 → Hydra 覆盖 | `key=value ...`     | 否    | 空     | 将键值对传递给 Hydra，覆盖任意配置字段                 |
| discover | `--platform`          | CLI 选项            | 字符串                 | 否    | 配置文件值 | 指定运行时平台名，覆盖配置中的 `platform`             |
| discover | `--core-tables`       | CLI 选项            | 逗号分隔字符串             | 否    | 配置文件值 | 设定核心表集合，加速 Schema 分析                   |
| discover | `--max-tables`        | CLI 选项            | 整数                  | 否    | 配置文件值 | 限制参与分析的表数量                             |
| generate | `--workflow-config`   | CLI 选项            | 路径                  | 否    | 配置文件值 | 指向 `workflow_output/*.json`，为空时使用配置内默认 |
| generate | `--target-count`      | CLI 选项            | 整数                  | 否    | 配置文件值 | 目标题目数量，建议根据业务规模调整                      |
| generate | `--template-count`    | CLI 选项            | 整数                  | 否    | 配置文件值 | 模板数量，影响题目覆盖度                           |
| generate | `--task-type`         | CLI 选项            | `query`/`cud`/`all` | 否    | 配置文件值 | 控制题目类型（查询/增删改/混合）                      |
| generate | `--platform`          | CLI 选项            | 字符串                 | 否    | 配置文件值 | 可单独覆盖题目生成阶段的平台名称                       |
| factory  | `--raw-question-dir`  | CLI 选项            | 路径                  | 否    | 配置文件值 | 批量读取题目目录，优先级低于 `--raw-question-file`   |
| factory  | `--raw-question-file` | CLI 选项            | 路径                  | 否    | 配置文件值 | 指定单个 `*.jsonl` 题目文件                    |
| factory  | `--max-task-id`       | CLI 选项            | 整数                  | 否    | 配置文件值 | 生成任务的起始/上限 ID，用于避免与既有任务冲突              |

> 生效优先级：显式 CLI 选项（含 `--config` 提供的 Hydra 覆盖） > 对应配置文件中的值 > 代码中的硬编码默认值。

#### 推荐评测部署流程

1. **工作流发现**
   ```bash
   python3 cli.py --conf=zentao discover \
     --platform=zentao
   ```
    - 如需限制处理的表数量，可额外添加 `--max-tables=80`。

2. **题目生成**
   ```bash 
   # 采用 Cli 显示定义参数
   python3 cli.py --conf=zentao generate \
     --workflow-config="/Users/sundapeng/Project/nlp/webrlvr/table2task/workflow_output/ZenTao.json" \
     --target-count=1 \
     --template-count=1 \
     --task-type=query
   ``` 

   或者

   ```bash
   # 采用 Hydra 覆盖配置键值
   python3 cli.py --conf=zentao --config="task_generation.workflow_config=/Users/sundapeng/Project/nlp/webrlvr/table2task/workflow_output/ZenTao.json task_generation.target_count=1 task_generation.template_count=1 task_generation.task_type=query" generate
   ```

3. **任务构建**
   ```bash
   # 采用 Cli 显示定义参数
   python3 cli.py --conf=zentao factory \
     --raw-question-file="/Users/sundapeng/Project/nlp/webrlvr/table2task/questions_bank/ZenTao/20250908145252-generated_tasks_enhanced.jsonl" \
   ```

   或者

   ```bash
   # 采用 Hydra 覆盖配置键值
   python3 cli.py --conf=zentao --config="task_factory.raw_question_file=/Users/sundapeng/Project/nlp/webrlvr/table2task/questions_bank/ZenTao/20250908145252-generated_tasks_enhanced.jsonl" factory
   ```
    - 若提供目录 `--raw-question-dir`，将批量处理其中的所有题目文件。

## 题目难度分析工具

用于分析SQL题目的难度，通过多个维度进行综合评分

### 难度等级划分

- **简单**: ≤1.5分
- **较简单**: 1.5-2.5分
- **中等**: 2.5-3.5分
- **较困难**: 3.5-4.5分
- **困难**: >4.5分

详见 [README.md](analysis/README.md)

## 数据库支持

### 当前支持的数据库类型

目前框架支持以下数据库类型：

- **MySQL**: 使用 `mysql-connector-python` 驱动
- **PostgreSQL**: 使用 `psycopg2` 驱动

### 扩展其他数据库类型

如需连接其他类型的数据库，请基于 `helper/db_conn_factory.py` 中的架构进行扩展。

#### 1. 实现 DatabaseAdapter 子类

创建新的适配器类，继承 `DatabaseAdapter` 基类并实现所有抽象方法。

#### 2. 更新 DatabaseConnectionFactory

在 `DatabaseConnectionFactory` 类中添加对应的映射关系。

#### 3. 更新配置文件

在配置文件中指定新的数据库类型：

```yaml
database:
  type: "your_database"  # 指定新的数据库类型
  host: localhost
  port: 5432
  user: username
  password: password
  database: your_database
  # 其他必要的配置参数
```

#### 4. 安装依赖

确保在 `requirements.txt` 中添加对应的数据库驱动依赖：

```
your-database-driver==x.x.x
```