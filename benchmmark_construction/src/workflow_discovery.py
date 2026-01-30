#!/usr/bin/env python3
"""
Complete Hydra version workflow discovery script
Integrates all functionality from workflow_discovery.py using Hydra configuration management
"""
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import hydra
import mysql.connector
from dotenv import load_dotenv
from omegaconf import DictConfig
from openai import OpenAI

from helper.db_conn_factory import DatabaseConnectionFactory

# Load environment variables
load_dotenv()

TABLE_ANALYSIS_PROMPT = """
You are a database analyst expert. Analyze the following table structure and provide insights about its business purpose and potential relationships.

Table: {table_name}
Schema: {schema_json}

Please analyze:
1. What business entity or process does this table represent?
2. What are the key fields and their purposes?
3. What types of relationships might this table have with other tables?
4. What business operations would typically involve this table?

IMPORTANT CONSTRAINTS:
- ONLY analyze the provided table structure
- DO NOT assume the existence of other tables
- DO NOT reference common tables like 'users', 'authors', 'categories' unless they are explicitly mentioned in the schema
- Focus on the actual fields and structure provided

Return JSON:
{{
    "business_purpose": "description of what this table represents",
    "key_fields": ["list of important field names"],
    "potential_relationships": ["types of relationships this table might have"],
    "business_operations": ["operations that would use this table"]
}}
"""

RELATIONSHIP_ANALYSIS_PROMPT = """
You are a database analyst expert. Analyze the potential relationships between two tables based on their schemas and business purposes.

Table 1: {table1_name}
Schema: {schema1_json}
Analysis: {analysis1_json}

Table 2: {table2_name}
Schema: {schema2_json}
Analysis: {analysis2_json}

IMPORTANT CONSTRAINTS:
- ONLY analyze relationships between the TWO tables provided above
- DO NOT reference any other tables that are not explicitly provided
- DO NOT assume the existence of common tables like 'users', 'authors', 'categories', etc.
- ONLY use fields that actually exist in the provided table schemas
- If no meaningful relationship exists between these two tables, return an empty relationships array

Based on the table structures and business purposes, suggest possible JOIN relationships. Consider:
1. Naming conventions and field similarities
2. Business logic and domain knowledge
3. Data types compatibility
4. Common database design patterns
5. Potential foreign key relationships (even if not explicitly defined)

For each suggested relationship, provide:
1. The JOIN condition
2. Confidence level (high/medium/low)
3. Reasoning for the relationship
4. Relationship type (foreign_key, business_logic, statistical, etc.)
5. A test SQL query to verify the relationship

Return JSON:
{{
    "relationships": [
        {{
            "type": "inferred",
            "from_table": "{table1_name}",
            "from_column": "field_name",
            "to_table": "{table2_name}",
            "to_column": "field_name",
            "condition": "{table1_name}.field = {table2_name}.field",
            "confidence": "high|medium|low",
            "reasoning": "explanation of why this relationship makes sense",
            "relationship_type": "foreign_key|business_logic|statistical|logical",
            "test_sql": "SELECT COUNT(*) FROM {table1_name} t1 JOIN {table2_name} t2 ON t1.field = t2.field LIMIT 1",
            "verified": false
        }}
    ]
}}
"""

WORKFLOW_GENERATION_PROMPT = """
You are a database analyst and business process expert. Based on the provided table analysis and relationships, generate comprehensive workflow configurations.

### Website Context
{website_context}

### Task
Generate {min_count}-{max_count} workflow configurations that represent meaningful business processes. Each workflow should group related tables that work together to accomplish a specific business goal.
For each workflow:
1. **Workflow Name**: Use descriptive English name (e.g., "user_registration", "order_processing", "content_management")
2. **Core Tables**: Include primary tables involved in the workflow ({min_tables_per_workflow}-{max_tables_per_workflow} tables per workflow)
    - Tables that are directly involved in the main business process
    - Tables that form the primary workflow steps
    - Tables that are essential for the workflow to function
3. **Key Relationships**: MUST ONLY include relationships from the provided Table Relationships data。
4. **Business Value**: Describe the business purpose and value
5. **Workflow Description**: Explain what this workflow accomplishes
6. **Related Tables**: Include additional tables that are explicitly listed in the schema and have direct relationships with core_tables. DO NOT include any tables that are not present in the provided database schema.

### Table Business Analysis
{table_analysis}

### Table Relationships
{relationships}

### Output Format
Return JSON with a "workflows" array. Each workflow should contain:
- name: Descriptive workflow name in English
- description: What this workflow accomplishes
- core_tables: Array of all tables involved in this workflow (must have relationships with each other)
- key_relationships: Array of relationship objects from the provided relationships data (exact copies, no modification)
- business_value: Business purpose and value
- workflow_type: Type of workflow (e.g., "user_management", "order_processing", "content_management")

### IMPORTANT RULES:
- Only use relationships that are explicitly provided in the relationships data
- Copy relationship objects exactly as they appear in the provided data
- Do not modify any fields in the relationship objects
- Do not create new relationships or change existing ones
- All relationships in key_relationships must be exact copies from the provided relationships data

### Example Output
{{
    "workflows": [
        {{
            "name": "project_management",
            "description": "Complete project lifecycle management from initiation to delivery, including requirement management, task assignment, and progress tracking",
            "core_tables": [
                "zt_project",
                "zt_story",
                "zt_task"
            ],
            "key_relationships": [
                {{
                    "type": "inferred",
                    "from_table": "zt_project",
                    "from_column": "id",
                    "to_table": "zt_story",
                    "to_column": "project",
                    "condition": "zt_project.id = zt_story.project",
                    "confidence": "high",
                    "reasoning": "The 'project' field in the 'zt_story' table is likely a foreign key referencing the 'id' in the 'zt_project' table, representing a one-to-many relationship where each project can have multiple user stories.",
                    "relationship_type": "foreign_key",
                    "verified": false
                }},
                ...
            ],
            "business_value": "Streamlined project management workflow that enables teams to efficiently plan, execute, and deliver software projects with clear task assignments and progress tracking",
            "workflow_type": "project_management"
        }}
    ]
}}
"""


class HydraUniversalWorkflowDiscoverer:
    """Universal workflow discoverer using Hydra configuration"""

    def __init__(self, cursor, config: DictConfig):
        self.cursor = cursor
        self.config = config
        self.table_info = {}
        self.relationships = []
        self.verified_relationships = []
        self.table_analysis = {}

        # Read cache settings from configuration (compatible with original UNIVERSAL_CONFIG)
        self._cache_enabled = config.cache.enabled
        self._cache_ttl = config.cache.ttl
        self._max_relationships_per_table = config.discovery.relationships.max_per_table
        self._max_total_relationships = config.discovery.relationships.max_total
        self._confidence_levels = dict(config.discovery.confidence_levels)

        # Cache for performance
        self._schema_cache = {}
        self._analysis_cache = {}
        self._relationship_cache = {}
        self._cache_timestamp = {}

        self.db_adapter = DatabaseConnectionFactory.create_adapter(self.cursor, self.config)

    def reset_connection(self):
        try:
            if hasattr(self, 'cursor') and self.cursor:
                self.cursor.close()
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()

            # 重新建立连接
            self.conn, self.cursor = DatabaseConnectionFactory.create_connection(self.config).connect()
            if self.conn and self.cursor:
                self.db_adapter = DatabaseConnectionFactory.create_adapter(self.cursor, self.config)
                print("[RESET] Database connection reset successfully")
                return True
        except Exception as e:
            print(f"[ERROR] Failed to reset connection: {e}")
            return False

    def analyze_tables(self, core_tables: List[str]):
        """Analyze basic information of core tables"""
        print(f"Analyzing {len(core_tables)} core tables...")

        valid_core_tables = []
        for table in core_tables:
            try:
                # Get table schema
                columns_data = self.db_adapter.describe_table(table)
                columns = []
                for row in columns_data:
                    columns.append({
                        "field": self.db_adapter.handle_bytes_data(row[0]),
                        "type": self.db_adapter.handle_bytes_data(row[1]),
                        "key": self.db_adapter.handle_bytes_data(row[3])
                    })

                # Get row count with timeout protection
                row_count = self.db_adapter.get_row_count(table)

                if row_count == 0:
                    print(f"{table}: 0 rows, skipped")
                    continue

                self.table_info[table] = {
                    "columns": columns,
                    "row_count": row_count
                }
                valid_core_tables.append(table)
                print(f"[SUCCESS] {table}: {len(columns)} columns, {row_count} rows")

            except mysql.connector.Error as err:
                # print(f"[ERROR] Error analyzing {table}: {err}")
                # Add empty entry to avoid KeyError later
                # self.table_info[table] = {
                #     "columns": [],
                #     "row_count": 0
                # }
                if "Lost connection" in str(err) or "timeout" in str(err).lower():
                    print(f"[RESET] Connection lost for {table}, attempting reset...")
                    reset_status = self.reset_connection()
                    if reset_status:
                        print(f"[SKIP] {table}: Skipping after connection reset")
                        continue
                    else:
                        print(f"[FATAL] Cannot reset connection, stopping analysis")
                        break
                else:
                    print(f"[ERROR] {table}: {err}, skipping")
                    continue

        return valid_core_tables

    def _create_table_pairs(self, tables):
        """Create all table pairs that need analysis"""
        pairs = []
        for i, table1 in enumerate(tables):
            for table2 in tables[i + 1:]:
                pairs.append((table1, table2))
        return pairs

    def discover_relationships_parallel(self, tables, client=None):
        """Discover table relationships using parallel analysis"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        print("Discovering table relationships using parallel analysis...")

        # Get schemas for all tables
        schemas = {}
        for table in tables:
            try:
                schema = self._get_detailed_table_schema(table)
                if "error" not in schema:
                    schemas[table] = schema
                    print(f"[SUCCESS] Got schema for {table}: {len(schema['columns'])} columns")
                else:
                    print(f"[ERROR] Error getting schema for {table}: {schema['error']}")
            except Exception as e:
                print(f"[ERROR] Exception getting schema for {table}: {e}")
                continue

        if not schemas:
            print("No valid schemas found")
            return []

        # Create table pairs
        table_pairs = self._create_table_pairs(tables)
        print(f"Created {len(table_pairs)} table pairs for analysis")

        # Set concurrency parameters
        max_workers = min(self.config.discovery.relationships.parallel, len(table_pairs))

        relationships = []
        start_time = time.time()

        def _analyze_table_pair_worker(args):
            """Worker function for analyzing a single table pair"""
            table1, table2, schemas, client, analyzer = args
            try:
                # Check foreign key relationships
                fk_relationships = analyzer._check_foreign_key_relationships(table1, table2, schemas)

                # If no foreign key relationships, use LLM analysis
                if not fk_relationships and client:
                    llm_relationships = analyzer._analyze_relationships_with_llm_with_client(
                        table1, table2, schemas, client
                    )
                    return fk_relationships + llm_relationships
                else:
                    return fk_relationships
            except Exception as e:
                print(f"[ERROR] Error analyzing {table1} <-> {table2}: {e}")
                return []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(_analyze_table_pair_worker, (table1, table2, schemas, client, self)): (
                    table1, table2)
                for table1, table2 in table_pairs
            }

            # Collect results
            completed = 0
            for future in as_completed(future_to_pair):
                table1, table2 = future_to_pair[future]
                completed += 1

                try:
                    result = future.result()
                    relationships.extend(result)
                    print(
                        f"[SUCCESS] Completed {completed}/{len(table_pairs)}: {table1} <-> {table2} ({len(result)} relationships)")
                except Exception as e:
                    print(f"[ERROR] Failed {table1} <-> {table2}: {e}")

        # Deduplicate and sort
        # unique_relationships = self._deduplicate_and_sort_relationships(relationships)
        unique_relationships = []
        seen = set()

        # Add timeout protection for large datasets
        max_relationships = self._max_total_relationships
        processed_count = 0

        for rel in relationships:
            if processed_count >= max_relationships:
                print(f"[WARNING] Reached maximum relationship limit ({max_relationships})")
                break

            try:
                key = (rel["from_table"], rel["from_column"], rel["to_table"], rel["to_column"])
                if key not in seen:
                    seen.add(key)
                    unique_relationships.append(rel)
                processed_count += 1
                
            except Exception as e:
                print(f"[WARNING] Skipping invalid relationship due to error: {e}")
                print(f"[DEBUG] Problematic relationship data: {rel}")
                continue

        # Sort by confidence
        confidence_order = self._confidence_levels
        unique_relationships.sort(key=lambda x: confidence_order.get(x.get("confidence", "low"), 0), reverse=True)

        self.relationships = unique_relationships

        elapsed_time = time.time() - start_time
        print(f"[SUCCESS] Parallel analysis completed in {elapsed_time:.2f}s")
        print(f"Discovered {len(unique_relationships)} unique relationships")

        return unique_relationships

    def _verify_table_exists(self, table_name: str) -> bool:
        """Lightweight verification: check if table exists"""
        try:
            self.cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            result = self.cursor.fetchone()
            return result is not None
        except Exception:
            return False

    def verify_relationships(self, discovered_relationships: List) -> List:
        """Verify discovered relationships by executing test SQL with caching."""
        print("Verifying discovered relationships...")

        verified_list = []
        for rel in discovered_relationships:

            # Field fallback to prevent KeyError
            if "condition" not in rel:
                rel["condition"] = "[unknown]"
            if "from_table" not in rel:
                rel["from_table"] = "[unknown]"
            if "to_table" not in rel:
                rel["to_table"] = "[unknown]"

            if not rel["verified"]:
                test_sql = rel.get("test_sql", "")
                if not test_sql:
                    from_table = self.db_adapter.format_table_name(rel['from_table'])
                    to_table = self.db_adapter.format_table_name(rel['to_table'])
                    test_sql = f"SELECT COUNT(*) FROM {from_table} JOIN {to_table} ON {rel['condition']} LIMIT 1"

                try:
                    self.cursor.execute(test_sql)
                    print(f"Executing test SQL: {test_sql}")
                    result = self.cursor.fetchone()

                    if result is not None:
                        if result[0] > 0:
                            rel["verified"] = True
                            rel["test_result"] = result[0]
                            rel["verification_method"] = "sql_test"
                            rel["confidence"] = "high"
                            verified_list.append(copy.deepcopy(rel))
                            print(
                                f"[SUCCESS] Verified relationship with SQL: {rel['from_table']} -> {rel['to_table']} (high confidence)")
                        else:
                            rel["verified"] = False
                            rel["test_result"] = f"SQL test {test_sql} executed successfully, but no matching data."
                            rel["verification_method"] = "sql_test"
                            print(
                                f"[WARNING] Verified relationship with SQL: {rel['from_table']} -> {rel['to_table']} (low confidence - no matching data)")
                    else:
                        rel["verified"] = False
                        rel["test_result"] = f"SQL test {test_sql} executed failed."
                        rel["verification_method"] = "sql_test"
                        print(f"[ERROR] Relationship verification failed: {rel['from_table']} -> {rel['to_table']}")

                except Exception as err:
                    print(f"[ERROR] SQL error during verification: {rel['from_table']} -> {rel['to_table']} - {err}")
                    rel["verified"] = False
                    rel["verification_method"] = "error"
                    rel["error"] = str(err)
                    # Ensure cleanup of any unprocessed result sets
                    try:
                        self.cursor.fetchall()
                    except:
                        pass
            else:
                verified_list.append(copy.deepcopy(rel))

        self.relationships = discovered_relationships
        self.verified_relationships = verified_list
        return self.verified_relationships

    def _get_detailed_table_schema(self, table: str) -> Dict:
        """Get detailed schema information for a table with proper type handling."""
        return self._fetch_schema_from_db(table)

    def _fetch_schema_from_db(self, table: str) -> Dict:
        """Fetch schema from database."""
        try:
            # Use adapter to get table structure
            columns_data = self.db_adapter.describe_table(table)
            columns = []
            for row in columns_data:
                columns.append({
                    "field": self.db_adapter.handle_bytes_data(row[0]),
                    "type": self.db_adapter.handle_bytes_data(row[1]),
                    "null": self.db_adapter.handle_bytes_data(row[2]),
                    "key": self.db_adapter.handle_bytes_data(row[3]),
                    "default": self.db_adapter.handle_bytes_data(row[4]) if len(row) > 4 else None,
                    "extra": self.db_adapter.handle_bytes_data(row[5]) if len(row) > 5 else ""
                })

            # Use adapter to get foreign keys
            foreign_keys_data = self.db_adapter.get_foreign_keys(table)
            foreign_keys = []
            for row in foreign_keys_data:
                foreign_keys.append({
                    "column": self.db_adapter.handle_bytes_data(row[0]),
                    "referenced_table": self.db_adapter.handle_bytes_data(row[1]),
                    "referenced_column": self.db_adapter.handle_bytes_data(row[2])
                })

            return {
                "table_name": table,
                "columns": columns,
                "foreign_keys": foreign_keys
            }

        except Exception as e:
            print(f"Error getting schema for {table}: {e}")
            return {"table_name": table, "error": str(e)}

    def _check_foreign_key_relationships(self, table1: str, table2: str, schemas: Dict) -> List[Dict]:
        """Check for explicit foreign key relationships between two tables."""
        relationships = []

        schema1 = schemas.get(table1, {})
        schema2 = schemas.get(table2, {})

        # Check foreign keys from table1 to table2
        for fk in schema1.get("foreign_keys", []):
            if fk["referenced_table"] == table2:
                relationships.append({
                    "type": "foreign_key",
                    "from_table": table1,
                    "from_column": fk["column"],
                    "to_table": table2,
                    "to_column": fk["referenced_column"],
                    "condition": f"{table1}.{fk['column']} = {table2}.{fk['referenced_column']}",
                    "confidence": "high",
                    "reasoning": "Explicit foreign key constraint",
                    "relationship_type": "foreign_key",
                    "verified": True
                })

        # Check foreign keys from table2 to table1
        for fk in schema2.get("foreign_keys", []):
            if fk["referenced_table"] == table1:
                relationships.append({
                    "type": "foreign_key",
                    "from_table": table2,
                    "from_column": fk["column"],
                    "to_table": table1,
                    "to_column": fk["referenced_column"],
                    "condition": f"{table2}.{fk['column']} = {table1}.{fk['referenced_column']}",
                    "confidence": "high",
                    "reasoning": "Explicit foreign key constraint",
                    "relationship_type": "foreign_key",
                    "verified": True
                })

        return relationships

    def _analyze_relationships_with_llm_with_client(self, table1: str, table2: str, schemas: Dict, client) -> List[
        Dict]:
        """Use LLM to analyze possible relationships between tables with client."""
        return self._fetch_llm_relationships(table1, table2, schemas, client)

    def _fetch_llm_relationships(self, table1: str, table2: str, schemas: Dict, client) -> List[Dict]:
        """Fetch relationships from LLM."""
        schema1 = schemas.get(table1, {})
        schema2 = schemas.get(table2, {})

        # Get table analysis if available
        analysis1 = self.table_analysis.get(table1, {})
        analysis2 = self.table_analysis.get(table2, {})

        prompt = RELATIONSHIP_ANALYSIS_PROMPT.format(
            table1_name=table1,
            schema1_json=json.dumps(schema1, indent=2),
            analysis1_json=json.dumps(analysis1, indent=2),
            table2_name=table2,
            schema2_json=json.dumps(schema2, indent=2),
            analysis2_json=json.dumps(analysis2, indent=2)
        )

        try:
            response = client.chat.completions.create(
                model=self.config.openai.model,
                messages=[
                    {"role": "system", "content": "You are a database analyst expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.openai.temperature,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            relationships = result.get("relationships", [])

            # Convert to our format
            formatted_relationships = []
            for rel in relationships:
                # Generate condition if not provided
                condition = rel.get("condition", "")
                if not condition and rel.get("from_column") and rel.get("to_column"):
                    condition = f"{rel.get('from_table', table1)}.{rel.get('from_column')} = {rel.get('to_table', table2)}.{rel.get('to_column')}"

                formatted_rel = {
                    "type": rel.get("type", "inferred"),
                    "from_table": rel.get("from_table", table1),
                    "from_column": rel.get("from_column", ""),
                    "to_table": rel.get("to_table", table2),
                    "to_column": rel.get("to_column", ""),
                    "condition": condition,
                    "confidence": rel.get("confidence", "low"),
                    "reasoning": rel.get("reasoning", ""),
                    "test_sql": rel.get("test_sql", ""),
                    "relationship_type": rel.get("relationship_type", "inferred"),
                    "verified": rel.get("verified", False)
                }
                formatted_relationships.append(formatted_rel)

            return formatted_relationships

        except Exception as e:
            print(f"Error analyzing relationships with LLM: {e}")
            return []

    def analyze_table_business_purposes(self, tables: List[str], client) -> Dict:
        """Analyze business purposes of tables using LLM."""
        print("Analyzing table business purposes with LLM...")

        table_analysis = {}

        for table in tables:
            # Check cache
            if not self._cache_enabled:
                analysis = self._fetch_table_analysis(table, client)
                if analysis:
                    table_analysis[table] = analysis
                continue

            cache_key = f"analysis_{table}"
            current_time = time.time()

            if (cache_key in self._analysis_cache and
                    cache_key in self._cache_timestamp and
                    current_time - self._cache_timestamp[cache_key] < self._cache_ttl):
                print(f"[CACHE] Using cached analysis for {table}")
                table_analysis[table] = self._analysis_cache[cache_key]
                continue

            analysis = self._fetch_table_analysis(table, client)
            if analysis:
                # Cache result
                self._analysis_cache[cache_key] = analysis
                self._cache_timestamp[cache_key] = current_time
                table_analysis[table] = analysis

                print(f"[SUCCESS] Analyzed {table}: {analysis.get('business_purpose', 'Unknown')}")

        self.table_analysis = table_analysis
        return table_analysis

    def _fetch_table_analysis(self, table: str, client) -> Dict:
        """Fetch table analysis from LLM."""
        try:
            schema = self._get_detailed_table_schema(table)
            if "error" in schema:
                return {}

            prompt = TABLE_ANALYSIS_PROMPT.format(
                table_name=table,
                schema_json=json.dumps(schema, indent=2)
            )

            response = client.chat.completions.create(
                model=self.config.openai.model,
                messages=[
                    {"role": "system", "content": "You are a database analyst expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.openai.temperature,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"[ERROR] Error analyzing {table}: {e}")
            return {}

    def generate_workflows_with_llm(self, client) -> List[Dict]:
        """Generate workflows using LLM with universal business module identification"""

        print("Generating workflows with LLM...")

        # Prepare context for LLM
        _table_analysis = copy.deepcopy(self.table_analysis)
        for _, v in _table_analysis.items():
            v.pop("potential_relationships", None)
            v.pop("business_operations", None)

        _verified_relationships = copy.deepcopy(self.verified_relationships)
        for _ in _verified_relationships:
            _.pop("reasoning", None)

        # potential_relationships
        table_analysis_str = json.dumps(_table_analysis, indent=2)
        relationships_str = json.dumps(_verified_relationships, indent=2)

        platform_info = self.config.conf.platform_info
        workflow_guidelines = self.config.conf.workflow_guidelines

        min_count_workflow = self.config.discovery.workflows.min_count
        max_count_workflow = self.config.discovery.workflows.max_count
        min_tables_per_workflow = self.config.discovery.workflows.min_tables_per_workflow
        max_tables_per_workflow = self.config.discovery.workflows.max_tables_per_workflow

        try:
            platform_info = '\n'.join([f"- {k}:{v}" for k, v in platform_info.items()])
        except:
            pass

        try:
            workflow_guidelines = '\n'.join([f"- {_}" for _ in workflow_guidelines])
        except:
            pass

        website_context = f"""
Platform: 
{platform_info}
Website Guidelines: 
{workflow_guidelines}
"""

        prompt = WORKFLOW_GENERATION_PROMPT.format(
            website_context=website_context,
            table_analysis=table_analysis_str,
            relationships=relationships_str,
            min_count=min_count_workflow,
            max_count=max_count_workflow,
            min_tables_per_workflow=min_tables_per_workflow,
            max_tables_per_workflow=max_tables_per_workflow
        )

        try:
            response = client.chat.completions.create(
                model=self.config.openai.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.openai.temperature,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            workflows = result.get("workflows", [])

            print(f"[SUCCESS] LLM generated {len(workflows)} workflows")
            for workflow in workflows:
                print(f"  - {workflow.get('name', 'Unknown')}: {len(workflow.get('core_tables', []))} tables")

            return workflows

        except Exception as e:
            print(f"Error generating workflows with LLM: {e}")
            raise e


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function using Hydra configuration"""
    output_base_dir = Path(cfg.output_base_dir)
    if not cfg.platform and not cfg.conf.platform_info.name:
        print("[ERROR] Error: Please specify platform name")
        return
    platform = cfg.platform or cfg.conf.platform_info.name
    core_tables = cfg.core_tables
    if isinstance(core_tables, str):
        core_tables = [table.strip() for table in core_tables.split(',')]
    elif isinstance(core_tables, (list, tuple)):
        core_tables = list(core_tables)
    else:
        core_tables = list(core_tables) if core_tables else []

    core_tables = [table for table in core_tables if table.strip()]

    output_dir = "workflow_output"
    output_dir = Path.joinpath(Path(output_base_dir), output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path.joinpath(output_dir, f"{platform}.json")
    if output_file.exists():
        print(f"[WARNING] Output file {output_file} already exists")
    else:
        print(f"[START] Starting universal workflow discovery")
        print(f"[INFO] Platform: {platform}")
        print(f"[INFO] Database: {cfg.conf.database.host}:{cfg.conf.database.port}/{cfg.conf.database.database}")
        print(f"[INFO] LLM Model: {cfg.openai.model}")
        print(f"[INFO] Cache: {'Enabled' if cfg.cache.enabled else 'Disabled'} (TTL: {cfg.cache.ttl}s)")
        print(f"[INFO] Output file: {output_file}")

        # Connect to database
        try:
            conn, cursor = DatabaseConnectionFactory.create_connection(cfg).connect()
            if not conn:
                print("[ERROR] Database connection failed, exiting program")
                return
            print("[SUCCESS] Database connection successful")

        except mysql.connector.Error as err:
            print(f"[ERROR] Database connection failed: {err}")
            return

        # Create OpenAI client
        try:
            if not cfg.openai.api_key:
                print("[ERROR] Error: Please set OPENAI_API_KEY environment variable")
                return

            client = OpenAI(
                api_key=cfg.openai.api_key,
                base_url=cfg.openai.base_url,
                timeout=cfg.openai.timeout
            )
            print("[SUCCESS] OpenAI client created successfully")

        except Exception as err:
            print(f"[ERROR] OpenAI client creation failed: {err}")
            return

        try:
            # Create discoverer instance
            discoverer = HydraUniversalWorkflowDiscoverer(cursor, cfg)

            max_tables = cfg.discovery.max_tables
            if len(core_tables) == 0:
                core_tables = discoverer.db_adapter.get_all_table_names()

            # Execute discovery process
            print("\n" + "=" * 50)
            core_tables = discoverer.analyze_tables(core_tables)
            core_tables.sort()

            if len(core_tables) > max_tables:
                raise ValueError(f"Input table count ({len(core_tables)}) exceeds maximum limit ({max_tables})")
            print(f"Database tables: {len(core_tables)} tables")

            core_tables_str = ','.join(core_tables)
            core_tables_md5_hash = hashlib.md5(core_tables_str.encode('utf-8')).hexdigest()

            business_analysis_dir = Path.joinpath(output_dir, "business_analysis")
            business_analysis_dir.mkdir(parents=True, exist_ok=True)

            business_analysis_file = business_analysis_dir / f"{platform}_business_analysis_{core_tables_md5_hash}.json"

            if business_analysis_file.exists():
                with open(business_analysis_file, "r") as f:
                    business_analysis = json.load(f)
                    table_analysis = business_analysis.get("table_analysis", None)
                    relationships = business_analysis.get("relationships", None)

                    print("\n" + "=" * 50)
                    print("Load table_analysis from cache file.")
                    discoverer.table_analysis = table_analysis

                    print("\n" + "=" * 50)
                    print("Load relationships from cache file.")
                    discoverer.relationships = relationships
                    discoverer.verified_relationships = relationships
            else:
                print("\n" + "=" * 50)
                table_analysis = discoverer.analyze_table_business_purposes(core_tables, client)

                print("\n" + "=" * 50)
                relationships = discoverer.discover_relationships_parallel(core_tables, client)

                print("\n" + "=" * 50)
                relationships = discoverer.verify_relationships(relationships)

                with open(business_analysis_file, "w") as f:
                    f.write(json.dumps({
                        "core_tables": core_tables,
                        "table_analysis": table_analysis,
                        "relationships": relationships
                    }, ensure_ascii=False, indent=4))

            print("\n" + "=" * 50)
            workflows = discoverer.generate_workflows_with_llm(client)

            output = {
                "platform": platform,
                "core_tables": core_tables,
                "discovered_workflows": {},
                "discovery_metadata": {
                    "total_tables_analyzed": len(core_tables),
                    "relationships_found": len(relationships),
                    "workflows_identified": len(workflows),
                    "table_analysis_count": len(table_analysis),
                    "config_used": {
                        "discovery_type": "universal",
                        "llm_model": cfg.openai.model,
                        "max_relationships": cfg.discovery.relationships.max_total,
                        "confidence_levels": dict(cfg.discovery.confidence_levels)
                    }
                }
            }

            for i, workflow in enumerate(workflows):
                # Generate English key from workflow name
                workflow_name = workflow.get("name", f"workflow_{i}")
                workflow_key = workflow_name.lower().replace(" ", "_").replace("-", "_").replace("_workflow", "")
                output["discovered_workflows"][workflow_key] = workflow

            # Save results
            print(f"\n[SAVE] Saving results to: {output_file}")

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            print(f"\n[SUCCESS] Workflow discovery completed!")
            print(f"[INFO] Analyzed {len(core_tables)} tables")
            print(f"[INFO] Discovered {len(relationships)} relationships")
            print(f"[INFO] Generated {len(workflows)} workflows")
            print(f"[INFO] Results saved to: {output_file}")
            print(
                f"[INFO] Configuration used: {cfg.conf.database.host}:{cfg.conf.database.port}/{cfg.conf.database.database} + {cfg.openai.model}")

        finally:
            if conn:
                db_adapter = DatabaseConnectionFactory.create_adapter(cursor, cfg)
                if db_adapter.is_connection_active(conn):
                    cursor.close()
                    conn.close()


if __name__ == "__main__":
    main()
