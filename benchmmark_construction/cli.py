# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Authors :       sundapeng.sdp
   Date：          2025/9/8
   Description :
-------------------------------------------------
"""
__author__ = 'sundapeng.sdp'

import click
import hydra

from src.generate_questions import main as generate_questions_main
from src.task_factory import main as task_factory_main
from src.workflow_discovery import main as workflow_discovery_main


@click.group()
@click.option('--config-path', default='config', help='配置文件路径')
@click.option('--config-name', default='config', help='配置文件名')
@click.option('--conf', required=True, help='配置组名称，如zentao、espocrm等')
@click.option('--config', help='配置参数格式:key1=value1 key2=value2')
@click.pass_context
def cli(ctx, config_path, config_name, conf, config):
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config_path
    ctx.obj['config_name'] = config_name
    ctx.obj['conf'] = conf

    if config:
        ctx.obj['config'] = config.split()
    else:
        ctx.obj['config'] = []
    print(f"Debug: config = {ctx.obj['config']}")


@cli.command()
@click.option('--platform', help='平台名称')
@click.option('--core-tables', help='核心表列表，逗号分隔')
@click.option('--max-tables', type=int, help='最大表数量')
@click.pass_context
def discover(ctx, platform, core_tables, max_tables):
    with hydra.initialize(config_path=ctx.obj['config_path'], version_base=None):
        overrides = list(ctx.obj['config'])
        overrides.append(f'conf={ctx.obj["conf"]}')
        if platform:
            overrides.append(f'platform={platform}')
        if core_tables:
            overrides.append(f'core_tables={core_tables}')
        if max_tables:
            overrides.append(f'discovery.max_tables={max_tables}')

        cfg = hydra.compose(config_name=ctx.obj['config_name'], overrides=overrides)
        workflow_discovery_main(cfg)


@cli.command()
@click.option('--target-count', type=int, help='目标题目数量')
@click.option('--template-count', type=int, help='模板数量')
@click.option('--task-type', type=click.Choice(['query', 'cud', 'all']), help='任务类型')
@click.option('--workflow-config', help='工作流配置文件')
@click.option('--platform', help='平台名称')
@click.pass_context
def generate(ctx, target_count, template_count, task_type, workflow_config, platform):
    with hydra.initialize(config_path=ctx.obj['config_path'], version_base=None):
        overrides = list(ctx.obj['config'])
        overrides.append(f'conf={ctx.obj["conf"]}')
        if target_count:
            overrides.append(f'task_generation.target_count={target_count}')
        if template_count:
            overrides.append(f'task_generation.template_count={template_count}')
        if task_type:
            overrides.append(f'task_generation.task_type={task_type}')
        if workflow_config:
            overrides.append(f'task_generation.workflow_config={workflow_config}')
        if platform:
            overrides.append(f'platform={platform}')

        cfg = hydra.compose(config_name=ctx.obj['config_name'], overrides=overrides)
        generate_questions_main(cfg)


@cli.command()
@click.option('--raw-question-dir', help='原始题目目录')
@click.option('--raw-question-file', help='原始题目文件')
@click.option('--max-task-id', type=int, help='最大任务ID')
@click.pass_context
def factory(ctx, raw_question_dir, raw_question_file, max_task_id):
    with hydra.initialize(config_path=ctx.obj['config_path'], version_base=None):
        overrides = list(ctx.obj['config'])
        overrides.append(f'conf={ctx.obj["conf"]}')
        if raw_question_dir:
            overrides.append(f'task_factory.raw_question_dir={raw_question_dir}')
        if raw_question_file:
            overrides.append(f'task_factory.raw_question_file={raw_question_file}')
        if max_task_id:
            overrides.append(f'task_factory.max_task_id={max_task_id}')

        cfg = hydra.compose(config_name=ctx.obj['config_name'], overrides=overrides)
        task_factory_main(cfg)


if __name__ == '__main__':
    cli()
