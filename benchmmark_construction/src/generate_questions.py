import enum
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import hydra
import mysql.connector
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

from helper.db_conn_factory import DatabaseConnectionFactory

# --- Configuration ---
load_dotenv()

server_address = "127.0.0.1"


class DifficultyLevel(enum.Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class EnhancedTaskGenerator:
    def __init__(self, client, cursor, config: DictConfig, custom_config_file=None):
        self.client = client
        self.cursor = cursor
        self.config = config
        self.task_type = "query"  # Default to query-only tasks

        # Pre-discovered relationship storage
        self.pre_discovered_relationships = {}  # Store pre-discovered relationships
        # Try to load custom config first, then fall back to default
        if custom_config_file and os.path.exists(custom_config_file):
            self.platform_config = self._load_custom_config(custom_config_file)
        else:
            raise ValueError("Invalid platform name or custom config file")

        self.website_name = self.platform_config["platform"]

        # Cache related
        self._schema_cache = {}  # Table schema cache
        self._relationship_cache = {}  # Relationship analysis cache
        self._verification_cache = {}  # Relationship verification cache
        self._cache_timestamp = {}  # Cache timestamp
        self._cache_ttl = 3600  # Cache TTL (seconds)

        self.db_adapter = DatabaseConnectionFactory.create_adapter(self.cursor, self.config)

    def _preprocess_foreign_key_data(self, template: Dict, coordinated_data: Dict) -> (Dict, Dict):
        """Preprocess foreign key data to get actual IDs before SQL generation."""
        placeholder_values = {}
        placeholder_values_data_type = {}
        for placeholder_info in template.get("placeholders", []):
            placeholder_name = placeholder_info.get("name", "")
            table_name = placeholder_info.get("table", "")
            field_name = placeholder_info.get("field", "")
            field_type = placeholder_info.get("field_type", "")

            # Find corresponding data
            if table_name in coordinated_data and field_name in coordinated_data[table_name]:
                value = coordinated_data[table_name][field_name]
                formatted_placeholder_name = placeholder_name.replace(".", "-")
                placeholder_values[formatted_placeholder_name] = value
                placeholder_values_data_type[formatted_placeholder_name] = field_type

                # If it's a foreign key field, need to get corresponding ID
                if field_name == "id":
                    # This is an ID field, use directly
                    id_placeholder_name = f"{table_name}-id"
                    placeholder_values[id_placeholder_name] = value
                else:
                    # This is a name field, need to query corresponding ID
                    if value and value != "" and value is not None:  # Only query ID for non-empty values
                        try:
                            # Use adapter to format table name
                            formatted_table = self.db_adapter.format_table_name(table_name)
                            id_query = f"SELECT id FROM {formatted_table} WHERE {self.db_adapter.format_field_name(field_name)} = %s LIMIT 1"
                            self.cursor.execute(id_query, (value,))
                            id_result = self.cursor.fetchone()
                            if id_result:
                                id_placeholder_name = f"{table_name}-id"
                                placeholder_values[id_placeholder_name] = id_result[0]
                                print(f"Found ID for {table_name}.{field_name}={value}: {id_result[0]}")
                            else:
                                print(f"Warning: No ID found for {table_name}.{field_name}={value}")
                        except Exception as e:
                            print(f"Error querying ID for {table_name}.{field_name}: {e}")
                    else:
                        print(f"Warning: Empty value for {table_name}.{field_name}, will be handled by LLM adjustment")

                    print(f"Filled {formatted_placeholder_name} with {value}")
            else:
                # If no data found, set empty value for LLM to handle
                formatted_placeholder_name = placeholder_name.replace(".", "-")
                placeholder_values[formatted_placeholder_name] = ""
                print(f"Warning: No data found for {table_name}.{field_name}, set to empty for LLM adjustment")

        return placeholder_values, placeholder_values_data_type

    def _format_current_values_for_prompt(self, placeholder_values: Dict, placeholder_values_data_type: Dict) -> str:
        """Format current values for prompt."""
        formatted = []
        for key, value in placeholder_values.items():
            data_type = placeholder_values_data_type.get(key, "")
            status = "EMPTY" if not value or value == "" else "HAS_VALUE"
            formatted.append(f"- {key}: '{value}' (Data Type:{data_type})({status})")
        return "\n".join(formatted)

    def _clear_unread_results(self):
        """Clear any unread results from the cursor to prevent 'Unread result found' errors."""
        try:
            self.cursor.fetchall()
        except:
            pass

    def _get_available_data_for_association_fields(self, template: Dict) -> Dict:
        """Get available data for association fields (dynamically based on placeholders)"""
        available_data = {}

        # Iterate through placeholders to find association fields
        for placeholder in template.get("placeholders", []):
            table_name = placeholder.get("table", "")
            field_name = placeholder.get("field", "")

            # Skip ID fields
            if field_name == "id":
                continue

            # Check if it's an association field
            if self._is_association_field(placeholder, template):
                try:
                    formatted_table = self.db_adapter.format_table_name(table_name)
                    formatted_field = self.db_adapter.format_field_name(field_name)

                    # Use adapter to get random ordering syntax
                    order_clause = self.db_adapter.get_random_order_clause()

                    # Query actual values for this field (randomly select 5)
                    query = f"SELECT {formatted_field} FROM {formatted_table} WHERE {formatted_field} IS NOT NULL {order_clause} LIMIT 5"
                    self.cursor.execute(query)
                    results = [row[0] for row in self.cursor.fetchall()]

                    if results:
                        key = f"{table_name}_{field_name}"
                        available_data[key] = results
                        print(f"Found {len(results)} available {field_name} values for {table_name}")
                    else:
                        print(f"Warning: No available {field_name} values found for {table_name}")

                except Exception as e:
                    print(f"Error getting available data for {table_name}.{field_name}: {e}")
                    self._clear_unread_results()

        return available_data

    def _is_association_field(self, placeholder: Dict, template: Dict) -> bool:
        """Determine if it's an association field"""
        table_name = placeholder.get("table", "")
        field_name = placeholder.get("field", "")

        # Rule 1: Not an ID field
        if field_name == "id":
            return False

        # Rule 2: Has corresponding ID field in placeholders
        has_corresponding_id = any(
            p.get("table") == table_name and p.get("field") == "id"
            for p in template.get("placeholders", [])
        )

        # Rule 3: Referenced in used_relationships
        is_referenced = any(
            r.get("to_table") == table_name
            for r in template.get("used_relationships", [])
        )

        # Rule 4: Field name indicates association field (name, realname, title, etc.)
        is_association_by_name = field_name in ['name', 'realname', 'title', 'account', 'code']

        return has_corresponding_id or is_referenced or is_association_by_name

    def _format_available_data_for_prompt(self, available_data: Dict) -> str:
        """Format available data for prompt"""
        if not available_data:
            return "No available data found."

        formatted = []
        for key, values in available_data.items():
            table_field = key.replace("_", ".")
            formatted.append(f"- {table_field}: {', '.join(values)}")

        return "\n".join(formatted)

    def _load_custom_config(self, config_file: str) -> Dict:
        """Load custom workflow configuration from a Python file or JSON file."""
        return self._load_json_config(config_file)

    def _should_filter_question(self, question: Dict) -> bool:
        question_text = question.get('question', '')

        # Filter questions containing empty strings
        if "''" in question_text or '""' in question_text:
            print(f"Filtered: Question contains empty string - {question_text[:50]}...")
            return True

        # 2. Check placeholder values
        placeholder_values = question.get('placeholder_values', {})
        for key, value in placeholder_values.items():
            if value == "" or value is None:
                print(f"Filtered: Placeholder {key} has empty value")
                return True

        # 3. Check for empty string conditions in SQL
        sql = question.get('sql', '')
        if "= ''" in sql or "= ''" in sql:
            print(f"Filtered: SQL contains empty string condition - {sql[:50]}...")
            return True

        # 4. 新增：检查 CUD 操作是否真正有效
        operation_type = question.get('operation_type', 'SELECT')
        if operation_type in ['INSERT', 'UPDATE', 'DELETE']:
            execution_result = question.get('execution_result', {})
            if execution_result:
                before_value = execution_result.get('before_execute_verification_value')
                after_value = execution_result.get('after_execute_verification_value')

                # 检查 DELETE 操作是否真正删除了记录
                if operation_type == 'DELETE' and before_value == after_value:
                    print(f"Filtered: DELETE operation had no effect - before: {before_value}, after: {after_value}")
                    return True

                # 检查 UPDATE 操作是否真正更新了记录
                if operation_type == 'UPDATE' and before_value == after_value:
                    print(f"Filtered: UPDATE operation had no effect - before: {before_value}, after: {after_value}")
                    return True

                # 检查 INSERT 操作是否真正插入了记录
                if operation_type == 'INSERT' and after_value == 0:
                    print(f"Filtered: INSERT operation failed - no records created")
                    return True

        return False

    def _get_pre_discovered_relationships(self, table1: str, table2: str) -> List[Dict]:
        """Get pre-discovered relationships"""
        relationship_key = f"{table1}_{table2}"
        reverse_key = f"{table2}_{table1}"

        # Check forward and reverse relationships
        if relationship_key in self.pre_discovered_relationships:
            return self.pre_discovered_relationships[relationship_key]
        elif reverse_key in self.pre_discovered_relationships:
            return self.pre_discovered_relationships[reverse_key]

        return []

    def _analyze_relationships_with_llm(self, table1: str, table2: str, schemas: Dict) -> List[Dict]:
        """Use LLM to analyze possible relationships between tables."""

        schema1 = schemas.get(table1, {})
        schema2 = schemas.get(table2, {})

        prompt = f"""
Analyze the database schema and suggest possible JOIN relationships between two tables.

Table 1: {table1}
Schema: {json.dumps(schema1, indent=2)}

Table 2: {table2}
Schema: {json.dumps(schema2, indent=2)}

Suggest possible JOIN conditions based on:
1. Naming conventions (e.g., table_id, table_name)
2. Business logic and domain knowledge
3. Data types compatibility
4. Common patterns in database design

For each suggested relationship, provide:
1. The JOIN condition
2. Confidence level (high/medium/low)
3. Reasoning for the relationship
4. A test SQL query to verify the relationship

Return JSON:
{{
    "relationships": [
        {{
            "type": "inferred",
            "from_table": "{table1}",
            "from_column": "field_name",
            "to_table": "{table2}",
            "to_column": "field_name",
            "condition": "{table1}.field = {table2}.field",
            "confidence": "high|medium|low",
            "reasoning": "explanation",
            "test_sql": "SELECT COUNT(*) FROM {table1} t1 JOIN {table2} t2 ON t1.field = t2.field LIMIT 1",
            "verified": false
        }}
    ]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.openai.model,
                messages=[
                    {"role": "system", "content": "You are a database analyst expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            return result.get("relationships", [])
        except Exception as e:
            print(f"Error analyzing relationships with LLM: {e}")
            return []

    def _get_operation_types_for_task_type(self) -> List[str]:
        """Get operation types based on task_type."""
        if self.task_type == "query":
            return ["SELECT"]
        elif self.task_type == "cud":
            return ["INSERT", "UPDATE", "DELETE"]
        elif self.task_type == "all":
            return ["SELECT", "INSERT", "UPDATE", "DELETE"]
        else:
            return ["SELECT"]  # Default to query

    def _load_json_config(self, json_file: str) -> dict:
        """Load workflow configuration from a JSON file with relationship support."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                platform_config = json.load(f)
                return platform_config
        except Exception as e:
            print(f"Error loading JSON config {json_file}: {e}")
            raise Exception(f"Error loading JSON config {json_file}: {e}")

    def get_configured_workflows(self) -> List[Dict]:
        """Get pre-configured workflows for the platform."""
        print(f"Loading configured workflows for {self.website_name}...")

        workflows = []
        platform_workflows = self.platform_config.get("discovered_workflows", {})

        for workflow_key, workflow_config in platform_workflows.items():
            workflows.append({
                **workflow_config, **{"key": workflow_key}
            })

        print(f"Loaded {len(workflows)} configured workflows:")
        for workflow in workflows:
            print(f"  - {workflow['name']}: {workflow['description']}")

        return workflows

    def generate_query_answer_for_eval(self, sql_result, question_text, sql=""):
        """
        Generate eval answer for query type questions
        """
        # Use LLM to generate accurate answer based on question, SQL and result
        return self.generate_answer_with_llm(sql_result, question_text, sql)

    def execute_and_validate_questions(self, questions: List[Dict]) -> List[Dict]:
        """Execute SQL queries and validate the results."""
        print("\n--- Executing SQL queries and validating results ---")

        validated_questions = []

        for question in questions:
            try:
                if self._should_filter_question(question):
                    continue

                operation_type = question.get('operation_type', 'SELECT')

                if operation_type == "SELECT":
                    # Handle query operations
                    sql = question.get('sql', '')
                    if not sql:
                        print(f"Warning: No SQL found for question: {question.get('question', 'Unknown')}")
                        continue

                    # Execute the SQL query
                    try:
                        # Execute the SQL query
                        self.cursor.execute(sql)
                        sql_result = self.cursor.fetchall()
                    except Exception as e:
                        print(f"SQL execution error: {e}")
                        print(f"Error question: {json.dumps(question, ensure_ascii=False)}")
                        try:
                            self.cursor.execute("ROLLBACK")
                        except:
                            pass
                        continue

                    # Generate answer for eval format
                    answer = self.generate_query_answer_for_eval(sql_result, question.get('question', ''), sql)

                    # Add execution results to question
                    question['answer'] = answer
                    question['sql_execute_result'] = sql_result

                    # Validate query result
                    if self._validate_query_result(question, len(sql_result)):
                        validated_questions.append(question)
                    else:
                        print(f"Filtering invalid query: {question['question'][:80]}...")

                else:
                    # Handle CUD operations
                    if self.task_type == "query":
                        print(f"Warning: Skipping CUD operation in query-only mode: {operation_type}")
                        continue

                    # Execute CUD operation with verification
                    execution_result = self.execute_cud_with_verification(question)

                    # Generate answer for CUD operation
                    answer = self.generate_cud_answer_for_eval(execution_result, question.get('question', ''))

                    # Add execution results to question
                    # after_execute_verification_value = execution_result.get("after_execute_verification_value", "")
                    # before_execute_verification_value = execution_result.get("before_execute_verification_value", "")
                    # if after_execute_verification_value == before_execute_verification_value:
                    #     raise Exception(f"Before and after execute verification values are the same for CUD operation: {operation_type}")

                    question['answer'] = [execution_result.get("after_execute_verification_value", "")]
                    question['execution_result'] = execution_result

                    # Only add successful CUD operations to validated questions
                    if execution_result.get('success', False):
                        validated_questions.append(question)
                        print(f"CUD operation successful: {operation_type}")
                    else:
                        print(
                            f"CUD operation failed: {operation_type} - {execution_result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"Error executing question: {e}")
                continue

        print(f"Validated {len(validated_questions)} questions")
        return validated_questions

    def _validate_query_result(self, question: Dict, row_count: int) -> bool:
        """Enhanced validation for query results."""
        # Basic checks
        if row_count == 0:
            return False  # No results for query

        if row_count > 100:
            return False  # Too many results

        # Check if the question makes sense for the result count
        question_text = question.get('question', '').lower()

        # For count queries, any number of results is fine
        if 'count' in question_text or 'how many' in question_text:
            return True

        # For list/show queries, reasonable number of results
        if 'list' in question_text or 'show' in question_text or 'find' in question_text:
            return 1 <= row_count <= 50

        # For single item queries, expect 1 result
        if 'first' in question_text or 'latest' in question_text or 'specific' in question_text:
            return row_count == 1

        # Default validation
        return 1 <= row_count <= 50

    def execute_cud_with_verification(self, question: Dict) -> Dict:
        """Execute CUD operation with improved verification and rollback tracking."""
        sql = question.get('sql', '')
        verification_sql = question.get('verification_sql', '')
        operation_type = question.get('operation_type', '')

        if not sql or not verification_sql:
            return {"success": False, "error": "Missing SQL or verification SQL"}

        try:
            # 1. Pre-execution state: Get initial verification SQL result
            print(f"Pre-execution: Checking initial state...")
            self.cursor.execute(verification_sql)
            pre_verification_result = self.cursor.fetchone()

            # Unified verification result processing
            before_execute_verification_value = pre_verification_result[0] if pre_verification_result else None
            before_execute_verification_count = 1 if pre_verification_result else 0

            print(f"Before execute verification value: {before_execute_verification_value}")

            # 2. Start transaction
            self.cursor.execute("START TRANSACTION")
            print(f"Started transaction for {operation_type} operation")

            # 3. Execute CUD operation
            sql_statements = [stmt.strip() for stmt in sql.split(';') if stmt.strip()]
            affected_rows_total = 0

            print(f"Executing {sql_statements} statements...")
            for i, statement in enumerate(sql_statements):
                if statement:
                    print(f"   Executing statement {i + 1}: {statement[:50]}...")
                    self.cursor.execute(statement)
                    affected_rows_total += self.cursor.rowcount

            # 4. Post-execution state: Get verification SQL result
            print(f"Post-execution: Checking final state...")
            self.cursor.execute(verification_sql)
            post_verification_result = self.cursor.fetchone()

            # Unified verification result processing
            after_execute_verification_value = post_verification_result[0] if post_verification_result else None
            after_execute_verification_count = 1 if post_verification_result else 0

            # 5. Determine if operation was successful
            if operation_type == "UPDATE":
                # For UPDATE, need to get expected new value from question
                expected_value = self._extract_expected_value_from_question(question)
                success = self._evaluate_cud_success(
                    operation_type,
                    before_execute_verification_value,
                    after_execute_verification_value,
                    expected_value
                )
            else:
                success = self._evaluate_cud_success(operation_type,
                                                     before_execute_verification_value,
                                                     after_execute_verification_value)

            if success:
                # 6a. Success: Record results then rollback transaction (ensure data consistency)
                print(f"CUD operation successful: {operation_type}")
                print(f"Changes would affect: {affected_rows_total} rows")
                print(f"Before: {before_execute_verification_count} records")
                print(f"After:  {after_execute_verification_count} records")

                # Rollback transaction to maintain data consistency
                self.cursor.execute("ROLLBACK")
                print(f"Rolling back successful operation to maintain data consistency")

                # Post-rollback state: Verify rollback took effect
                self.cursor.execute(verification_sql)
                rolled_back_verification_result = self.cursor.fetchone()
                rolled_back_verification_count = rolled_back_verification_result[
                    0] if rolled_back_verification_result else 0
                print(f"Rolled back verification count: {rolled_back_verification_count}")

                # Verify rollback was successful
                if rolled_back_verification_count == before_execute_verification_count:
                    print(f"Rollback successful: state restored to initial")
                else:
                    print(
                        f"Warning: Rollback may not be complete: {before_execute_verification_count} -> {rolled_back_verification_count}")

            else:
                # 6b. Failure: Rollback transaction
                self.cursor.execute("ROLLBACK")
                print(f"CUD operation failed: {operation_type}, rolling back")
                print(f"Changes rolled back: {affected_rows_total} rows would have been affected")

                # Post-rollback state: Verify rollback took effect
                self.cursor.execute(verification_sql)
                rolled_back_verification_result = self.cursor.fetchone()
                rolled_back_verification_count = rolled_back_verification_result[
                    0] if rolled_back_verification_result else 0
                print(f"Rolled back verification count: {rolled_back_verification_count}")

                # Verify rollback was successful
                if rolled_back_verification_count == before_execute_verification_count:
                    print(f"Rollback successful: state restored to initial")
                else:
                    print(
                        f"Warning: Rollback may not be complete: {before_execute_verification_count} -> {rolled_back_verification_count}")

                    # Unified result structure
            result = {
                "operation_type": operation_type,
                "before_execute_verification_value": before_execute_verification_value,
                "after_execute_verification_value": after_execute_verification_value,
                "success": success,
                "affected_rows": affected_rows_total
            }

            # For UPDATE operations, add expected value
            if operation_type == "UPDATE":
                result["expected_value"] = expected_value

            return result
        except Exception as e:
            # Exception handling: Ensure rollback
            self.cursor.execute("ROLLBACK")
            print(f"CUD operation error: {e}, rolling back")

            # Analyze error type
            error_msg = str(e)
            error_type = "unknown"

            if "Duplicate entry" in error_msg:
                error_type = "duplicate_entry"
                print(f"Duplicate entry detected - consider using unique value generation")
            elif "foreign key constraint" in error_msg.lower():
                error_type = "foreign_key_constraint"
                print(f"Foreign key constraint failed - check referenced records")
            elif "cannot be null" in error_msg.lower():
                error_type = "null_constraint"
                print(f"NULL constraint violated - check required fields")
            elif "data too long" in error_msg.lower():
                error_type = "data_length"
                print(f"Data too long for field - check field length limits")

            return {
                "success": False,
                "error": error_msg,
                "error_type": error_type,
                "operation_type": operation_type,
                "before_execute_verification_value": before_execute_verification_value if 'before_execute_verification_value' in locals() else None,
                "after_execute_verification_value": None
            }

    def _evaluate_cud_success(self, operation_type: str, before_value=None, after_value=None, expected_value=None) -> bool:
        """Evaluate if CUD operation was successful."""
        # Ensure verification_count is numeric type
        if operation_type == "INSERT":
            # INSERT successful: verification should return > 0
            after_count = int(after_value) if after_value is not None else 0
            before_count = int(before_value) if before_value is not None else 0
            return after_count > before_count
        elif operation_type == "UPDATE":
            # UPDATE successful: field value should change to expected value
            # Convert to string for comparison to avoid type mismatch
            before_str = str(before_value) if before_value is not None else ""
            after_str = str(after_value) if after_value is not None else ""
            expected_str = str(expected_value) if expected_value is not None else ""
            return after_str != before_str and after_str == expected_str
        elif operation_type == "DELETE":
            after_count = int(after_value) if after_value is not None else 0
            before_count = int(before_value) if before_value is not None else 0
            return before_count > after_count
        else:
            return False

    def _extract_expected_value_from_question(self, question: Dict) -> str:
        """Extract expected value from UPDATE question using simplified logic."""
        sql = question.get('sql', '')
        placeholder_values = question.get('placeholder_values', {})

        if not sql or not placeholder_values:
            return None

        try:
            # 找到 SET 的位置
            set_pos = sql.upper().find('SET')
            if set_pos == -1:
                return None

            # 找到 WHERE 或 JOIN 的位置（在 SET 之后）
            where_pos = sql.upper().find('WHERE', set_pos)
            join_pos = sql.upper().find('JOIN', set_pos)

            # SET 子句的结束位置
            end_pos = len(sql)
            if where_pos != -1:
                end_pos = where_pos
            if join_pos != -1:
                end_pos = min(end_pos, join_pos)

            # 提取 SET 子句
            set_clause = sql[set_pos:end_pos]

            # 匹配 SET field = 'value' 或 SET field = "value"
            set_match = re.search(r'SET\s+[^=]+\s*=\s*[\'"]([^\'"]+)[\'"]', set_clause, re.IGNORECASE)

            if set_match:
                value = set_match.group(1)
                # 检查是否是占位符格式 {table-field}
                if '{' in value and '}' in value:
                    placeholder_name = value.strip('{}')
                    if placeholder_name in placeholder_values:
                        return str(placeholder_values[placeholder_name])

                return str(value)

            return None

        except Exception as e:
            print(f"Error extracting expected value: {e}")
            return None

    def _generate_unique_value(self, table: str, field: str, base_value: str) -> str:
        """Generate unique value"""
        import time
        import random

        counter = 1
        new_value = base_value

        # Add timestamp to ensure uniqueness
        timestamp = int(time.time())
        random_suffix = random.randint(100, 999)

        while counter <= 10:  # Try up to 10 times
            try:
                sql = f"SELECT COUNT(*) FROM {table} WHERE {field} = %s"
                self.cursor.execute(sql, (new_value,))
                count = self.cursor.fetchone()[0]

                if count == 0:
                    return new_value

                # Generate new unique value
                new_value = f"{base_value}_{timestamp}_{random_suffix}_{counter}"
                counter += 1

            except Exception as e:
                print(f"Error checking uniqueness for {field}: {e}")
                self._clear_unread_results()
                return f"{base_value}_{timestamp}_{random_suffix}"

        # If 10 attempts fail, return value with timestamp
        return f"{base_value}_{timestamp}_{random_suffix}"

    def generate_cud_answer_for_eval(self, execution_result: Dict, question_text: str) -> List[str]:
        """Generate answer for CUD operations based on after_execute_verification_value."""
        if not execution_result.get("success", False):
            return ["0"]  # Operation failed

            # Unified use of after_execute_verification_value
            after_execute_verification_value = execution_result.get("after_execute_verification_value", 0)
            return [str(after_execute_verification_value)]

    def generate_intent_templates(self, workflow: Dict, verified_relationships: List,
                                  difficulty=DifficultyLevel.BASIC.value) -> List[Dict]:
        """Generate intent templates based on verified relationships with enhanced validation."""
        print(f"Generating intent templates with verified relationships for workflow: {workflow['name']}")

        # Prepare verification relationship information and table structure information
        relationship_info = json.dumps(verified_relationships, indent=2)
        table_schemas_info = self._format_table_schemas_for_llm(workflow['core_tables'])

        # Determine template types based on task_type
        operation_types = self._get_operation_types_for_task_type()

        platform_info = self.config.conf.platform_info
        workflow_guidelines = self.config.conf.workflow_guidelines

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

        prompt = f"""
Generate intent templates for a workflow using **verified table relationships** and **actual table schemas**.

## WEBSITE CONTEXT
{website_context}

## WORKFLOW CONTEXT
Workflow: {workflow['name']}
Description: {workflow['description']}
Core Tables: {workflow['core_tables']}
Task Type: {self.task_type}
Operation Types: {operation_types}
SQL Database Type: {self.config.conf.database.type}

## DATABASE SCHEMA
Table Schemas:
{table_schemas_info}

Verified Relationships:
{relationship_info}

---

## GOAL
Generate **{self.template_count} intent templates**.
The templates must be **exclusively** of type(s): **{operation_types}**.
The SQL must be compatible with **{self.config.conf.database.type}** database.

Each template must follow the structure and rules below.

---

## TEMPLATE STRUCTURE
- `template_type`: `"single_table"` or `"multi_table"`
- `operation_type`: Must be one of: {operation_types}
- `primary_table`: Exact table name
- `related_tables`: Related tables (empty if none)
- `placeholders`: Array of placeholder definitions, each containing:
  - `name`: Placeholder name (format: table-field)
  - `table`: Table name
  - `field`: Field name
  - `field_type`: Field data type from database schema (e.g., 'int', 'varchar(255)', 'datetime', 'boolean')
  - `description`: Description with expected data type
- `template`: English text with business context
- `zh_template`: Chinese text with business context
- `sql_template`: SQL query
- `verification_sql`: SQL used to verify the correctness of CUD operations.  
  - For **CUD (CREATE, UPDATE, DELETE)** operations: `verification_sql` should be simplified validation queries (e.g., `EXISTS`, `COUNT(*)`, or key field checks) that confirm whether the intended change has taken effect, while keeping the same filtering and join logic as in `sql_template`.  
  - For **SELECT (query)** operations: `verification_sql` must remain identical to `sql_template` (same structure, fields, and conditions) to ensure consistency.

---

## GLOBAL RULES
1. **Placeholders**
   - Format: `{{table-field}}`
   - Use exact table + field names (e.g. `zt_user-status`)
   - Use hyphen `-` instead of dot `.`
   - Must appear consistently in `template`, `zh_template`, and `sql_template`
   - CRITICAL: The description of each placeholder MUST clearly state its expected data type (e.g., 'Integer value', 'String for name', 'YYYY-MM-DD date') based on the database schema. This guides correct value input.

2. **SQL Rules & Data Type Integrity ({self.config.conf.database.type})**
   - **CRITICAL**: All table names and field names MUST be properly quoted according to {self.config.conf.database.type} syntax.
   - **{self.config.conf.database.type} Quoting Rules**:
     * MySQL: Use backticks (`) around table names and field names: `table_name` and `field_name`
     * PostgreSQL: Use double quotes (") around table names and field names: "table_name" and "field_name"
   - **Reserved Keyword Detection**: Check if table/field names are reserved keywords and quote them appropriately.
   - Always use table aliases (`t1`, `t2`, …).
   - Prefix all fields with the table alias.
   - STRING DATA: Must be enclosed in single quotes ('{{placeholder}}').
   - NUMERIC, BOOLEAN, or NULL DATA: Must NEVER be quoted ({{placeholder}}).
   - DATE/DATETIME DATA: Use the appropriate SQL function (e.g., NOW(), STR_TO_DATE(), DATE()). If a placeholder is used, ensure it matches the expected format and is quoted if the function requires a string argument.
   - No `SELECT *`.
   - Results must be **one verifiable value** only (COUNT, EXISTS, MAX, MIN, or a single field).
   - For multi-table queries, JOIN only via verified relationships.
   - WHERE conditions must filter by **specific fields with meaningful business attributes** (e.g., `status`, `name`, `version`), not just technical identifiers.
   - **Field Selection Rule**:  
     * DO NOT select `id` or pure foreign key fields as output.  
     * Always select business-descriptive fields (e.g., `username`, `status`, `osversion_name`, `type`, `created_at`).  
     * IDs may only be used internally for JOIN conditions, never in the final SELECT or placeholders.
   - The generated SQL MUST strictly adhere to the data types defined in the DATABASE SCHEMA for each field (e.g., INTEGER, VARCHAR, DATETIME, BOOLEAN).

3. **Verification Rules**
   - `SELECT`: same as `sql_template`
   - `INSERT`: verify record creation with COUNT
   - `UPDATE`: verify updated field value
   - `DELETE`: verify deletion with COUNT
   - Use JOINs if required by relationships

4. **Chinese Templates**
   - Natural language (not literal translation)
   - Use verbs: 查询, 获取, 统计, 创建, 更新, 删除
   - Proper Chinese sentence structure

5. **Field Rules**
   - Use only fields present in schema
   - For both single-table and multi-table queries:
        - IDs (like `id`, `project_id`, `user_id`) must NEVER appear in templates, placeholders, or descriptions.
        - They can be used **internally inside SQL subqueries or JOINs**, but the surface-level condition must always be expressed with a visible field (e.g., project `name`, user `realname`, story `title`).
    - Example (Bad): "...associated with project ID 2"
    - Example (Good): "...associated with project 'New Website Launch'"

6. **Relationship Connectivity**
   - All tables referenced by placeholders must be fully connected through the Verified Relationships.
   - If any placeholder table cannot be reached (directly or indirectly) via the Verified Relationships, the template must be discarded or rewritten.
   - SQL generation must only use paths that exist in Verified Relationships.

---

## TEMPLATE REQUIREMENTS
1. **Business Context Integration**
    - Templates must describe the intent from a **business perspective**, not technical schema terms.
    - Never mention internal IDs, codes, or version numbers.
      - *Bad*: “Find tasks in project ID 2”
      - *Good*: “Find tasks in project ‘Marketing Campaign 2025’”
    - Always use **user-facing fields** (e.g., project name, user full name, task title, status).
    - Multi-table templates must highlight a **business relationship** (e.g., “tasks assigned to users in department X”).
    - Single-table templates should focus on visible, meaningful attributes (e.g., status, deadlines, titles).

2. **Clarity of Expected Output**
    - The `template` and `zh_template` must **explicitly state what is being retrieved or measured**.
    - Avoid vague or underspecified intents.  
      - *Bad*: “As part of the IT operations team, I need to list all hypervisors related to farms with redundancy level 1.” (unclear what field or count is needed)  
      - *Good*: “As part of the IT operations team, I need to list the **names of all hypervisors** related to farms with redundancy level 1, so I can verify correct redundancy distribution.”  
      - *Good*: “As part of the IT operations team, I need to **count the number of hypervisors** related to farms with redundancy level 1, to ensure redundancy compliance.”  
    - Chinese templates must also follow this rule, e.g.,  
      - *Bad*: “作为IT运维团队的一员，我需要列出与冗余级别为1的农场相关的所有虚拟机管理程序。”  
      - *Good*: “作为IT运维团队的一员，我需要**列出与冗余级别为1的农场相关的所有虚拟机管理程序的名称**，以确保冗余分布正确。”  
      - *Good*: “作为IT运维团队的一员，我需要**统计与冗余级别为1的农场相关的虚拟机管理程序数量**，以确保冗余分布符合要求。”  

3. **Multi-Table Context Enhancement**
    - Enrich descriptions by **referencing related entities naturally**.
    - Always frame the request as a **real-world business scenario** tied to decision-making or analysis.
    - Avoid vague or abstract descriptions; ensure conditions map to **specific, visible fields** like names, statuses, roles, or dates.

---

## VISIBLE FIELD REQUIREMENTS
    **CRITICAL**: All questions must use ONLY visible/displayable fields that users can see and interact with in the web interface:
    Forbidden Patterns (NEVER use in templates/placeholders):
    - "查询ID为xxx的..."
    - "统计用户ID=xxx的..."
    - "检查ID=xxx是否存在..."
    - "WHERE id = xxx"
    - "project ID=xxx", "task ID=xxx", "story ID=xxx"

    Required Patterns:
    - "查询用户名为'xxx'的..."
    - "统计项目名为'xxx'的..."
    - "检查是否存在名为'xxx'的..."
    - "WHERE name = 'xxx'" or "WHERE realname = 'xxx'"
    - Always use visible fields such as `name`, `title`, `status`, `realname`, `code`.
    - Example: "associated with project '{{zt_project-name}}'" instead of "project ID 2"
    
---

## DIFFICULTY LEVELS
    The generated templates should **must** align with the {difficulty} level (flexible adherence is acceptable):
    - **basic**: Single-table query/operation with 1–2 placeholders  
    - **intermediate**: Involves 2–3 table JOINs with 2–3 placeholders  
    - **advanced**: Involves 3–5 table JOINs with 3–5 placeholders, conditions may include aggregation, time range, or AND/OR combinations  
    Notes:  
    - Placeholders must be based on **visible fields** (e.g., `name`, `title`, `status`, `realname`, `code`).  
    - IDs or *_id fields must **never** be used as placeholders.  
    - All placeholders should represent business-facing attributes that users can directly see and understand in the interface.  

---

## OPERATION REQUIREMENTS

### SELECT (Query)
Support multi-level query types:
- Level 1: Single Table**
    - COUNT, EXISTS, SINGLE, LIST, COMPARE, RANGE, NULL_CHECK, DISTINCT
- Level 2: Multi-Table**
    - JOIN_COUNT, JOIN_EXISTS, LEFT_JOIN, MULTI_JOIN, SELF_JOIN
- Level 3: Aggregation**
    - GROUP_COUNT, GROUP_AVG, GROUP_MAX, GROUP_SUM, HAVING, RANKING
- Level 4: Time**
    - TIME_RANGE, TIME_TREND, TIME_COMPARISON, RECENT_ACTIVITY, TIME_AGGREGATION

### INSERT (Create)
- Must be strictly **single-table operations** (no JOINs or multi-table INSERT).
- When generating INSERT statements, **consider schema constraints**:
    - Fields with `field_null = 'NO'` must be assigned a value (cannot be NULL)
    - Fields with `field_key = 'PRI'` or `field_default = None` must be explicitly set unless `auto_increment`
    - Fields with `field_extra` containing 'auto_increment' can be omitted  
- Generate valid insert queries with placeholders
- The VALUES clause must match the table's schema definition in number, order, and most importantly, DATA TYPE.
- No nested `SELECT` in VALUES
- `verification_sql` must confirm creation

### UPDATE (Modify)
- Must be strictly **single-table operations** (no JOINs or multi-table UPDATE).
- When updating, **consider schema constraints**:
    - Fields with `field_null = 'NO'` cannot be set to NULL
    - Primary key fields (`field_key = 'PRI'`) should generally not be updated
    - Fields without default values (`field_default = None`) must have valid assignment if involved  
- Must target specific records with WHERE
- The SET clause must assign values that are compatible with the data type of the target field.
- `verification_sql` must return updated field value

### DELETE (Remove)
- Must be strictly **single-table operations** (no JOINs or multi-table DELETE).
- When deleting, **consider schema constraints**:
    - Primary key fields (`field_key = 'PRI'`) are required in WHERE conditions to ensure precise deletion
    - Maintain referential integrity (avoid deleting rows referenced by other tables)  
- Maintain referential integrity
- `verification_sql` must confirm deletion

---

## EXAMPLES
### Single Table SELECT
```json
{{
  "template": "As a system administrator, I need to count how many users have status {{zt_user-status}} to monitor system access and security compliance",
  "zh_template": "作为系统管理员，我需要统计状态为{{zt_user-status}}的用户数量，用于监控系统访问和安全管理合规性",
  "sql_template": "SELECT COUNT(*) FROM zt_user t1 WHERE t1.status = '{{zt_user-status}}'",
  "verification_sql": "SELECT COUNT(*) FROM zt_user t1 WHERE t1.status = '{{zt_user-status}}'",
  "placeholders": [
    {{
      "name": "zt_user-status",
      "table": "zt_user",
      "field": "status",
      "field_type": "str",
      "description": "User status value"
    }}
  ]
}}
```

### Multi Table SELECT
```json
{{
  "template": "As a system administrator, I need to find all users who have access to project '{{zt_project-name}}' and are also members of team '{{zt_team-name}}', along with their roles and permissions to manage access control",
  "zh_template": "作为系统管理员，我需要查找所有有权访问项目'{{zt_project-name}}'且属于团队'{{zt_team-name}}'的用户，包括他们的角色和权限，以管理访问控制",
  "sql_template": "SELECT DISTINCT zt_user.realname as user_name, zt_user.email as user_email, zt_team.name as team_name, zt_project.name as project_name, zt_userrole.role as user_role, zt_userrole.permission as user_permission FROM zt_user JOIN zt_teamuser ON zt_user.account = zt_teamuser.user JOIN zt_team ON zt_teamuser.team = zt_team.id JOIN zt_userrole ON zt_user.account = zt_userrole.user JOIN zt_project ON zt_userrole.project = zt_project.id WHERE zt_project.name = '{{zt_project-name}}' AND zt_team.name = '{{zt_team-name}}' ORDER BY zt_user.realname ASC",
  "verification_sql": "SELECT DISTINCT zt_user.realname as user_name, zt_user.email as user_email, zt_team.name as team_name, zt_project.name as project_name, zt_userrole.role as user_role, zt_userrole.permission as user_permission FROM zt_user JOIN zt_teamuser ON zt_user.account = zt_teamuser.user JOIN zt_team ON zt_teamuser.team = zt_team.id JOIN zt_userrole ON zt_user.account = zt_userrole.user JOIN zt_project ON zt_userrole.project = zt_project.id WHERE zt_project.name = '{{zt_project-name}}' AND zt_team.name = '{{zt_team-name}}' ORDER BY zt_user.realname ASC",
  "placeholders": [
    {{
      "name": "zt_project-name",
      "table": "zt_project",
      "field": "name",
      "field_type": "str",
      "description": "Project name to check access for"
    }},
    {{
      "name": "zt_team-name",
      "table": "zt_team",
      "field": "name",
      "field_type": "str",
      "description": "Team name to filter users by"
    }}
  ]
}}
```

### FINAL OUTPUT

Return a JSON object:
{{
    "intent_templates": [
    // {self.template_count} templates here
]
}}
"""

        try:
            print(f"Starting template generation with enhanced prompts...")
            response = self.client.chat.completions.create(
                model=self.config.openai.model,
                messages=[
                    {"role": "system",
                     "content": "You are a database analyst expert. Generate templates with perfect placeholder consistency."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            templates = result.get("intent_templates", [])
            for template in templates:
                template["used_relationships"] = verified_relationships

            print(f"Successfully generated {len(templates)} original templates")
            return templates

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed, LLM may have returned incorrect format")
            return []

        except Exception as e:
            return []

    def _format_verified_relationships_for_llm(self, verified_relationships: List) -> str:
        """Format verified relationships for LLM prompt."""
        formatted = []

        for rel_key, relationships in verified_relationships.items():
            for rel in relationships:
                if rel.get("verified", False):
                    formatted.append(f"- {rel['condition']} (confidence: {rel['confidence']})")

        return "\n".join(formatted)

    def _format_table_schemas_for_llm(self, core_tables: List[str]) -> str:
        """Format table schemas for LLM prompt."""
        formatted = []

        for table in core_tables:
            try:
                # Use adapter to get table structure
                columns_data = self.db_adapter.describe_table(table)
                columns = []
                for row in columns_data:
                    field_name = self.db_adapter.handle_bytes_data(row[0])
                    field_type = self.db_adapter.handle_bytes_data(row[1])
                    field_null = self.db_adapter.handle_bytes_data(row[2])
                    field_key = self.db_adapter.handle_bytes_data(row[3])
                    field_default = self.db_adapter.handle_bytes_data(row[4])
                    field_extra = self.db_adapter.handle_bytes_data(row[5])
                    columns.append(f"  {field_name}: {field_type} [field_null={field_null} field_key={field_key} "
                                   f"field_default={field_default} field_extra={field_extra}]")

                table_schema = f"Table: {table}\n" + "\n".join(columns)
                formatted.append(table_schema)

            except Exception as e:
                print(f"Error getting schema for {table}: {e}")
                formatted.append(f"Table: {table} (Error: {e})")

        return "\n\n".join(formatted)

    def _build_coordinated_data_sql(self, primary_table: str, used_relationships: List[Dict], field_requirements: Dict,
                                    operation_type: str) -> str:
        """
        构建协调数据的大 SQL 查询，只包含模板涉及的表
        """
        # 构建 JOIN 子句
        joins = []
        select_fields = []

        # 添加主表字段
        for table_name, fields in field_requirements["primary_table_placeholder_fields"].items():
            for field in fields:
                select_fields.append(
                    f"{self.db_adapter.format_table_name(table_name)}.{self.db_adapter.format_field_name(field)}")

        # 添加关联表字段（只添加模板涉及的关联表）
        for table_name, fields in field_requirements["related_table_placeholder_fields"].items():
            for field in fields:
                select_fields.append(
                    f"{self.db_adapter.format_table_name(table_name)}.{self.db_adapter.format_field_name(field)}")

        # 构建表连接（只连接模板涉及的表）
        joins = self._build_template_table_connections(primary_table, field_requirements, used_relationships)

        if not joins:
            return None

        # 根据操作类型构建 WHERE 条件
        where_conditions = self._build_where_conditions_by_operation_type(primary_table, field_requirements,
                                                                          operation_type)

        # 构建完整的 SQL
        select_clause = ", ".join(select_fields)
        join_clause = " ".join(joins)
        where_clause = " AND ".join(where_conditions) if where_conditions else ""
        order_clause = self.db_adapter.get_random_order_clause()

        sql = (f"SELECT {select_clause} "
               f"FROM {self.db_adapter.format_table_name(primary_table)} "
               f"{join_clause} "
               f"{f'WHERE {where_clause}' if where_clause else ''} "
               f"{order_clause} "
               f"LIMIT 1")

        return sql

    def _build_where_conditions_by_operation_type(self, primary_table: str, field_requirements: Dict,
                                                  operation_type: str) -> List[str]:
        """
        根据操作类型构建 WHERE 条件
        """
        where_conditions = []

        if operation_type == "INSERT":
            # 插入类：除了主表之外的字段参与 WHERE 条件
            for table_name, fields in field_requirements["related_table_placeholder_fields"].items():
                for field in fields:
                    where_conditions.append(
                        f"{self.db_adapter.format_table_name(table_name)}.{self.db_adapter.format_field_name(field)} IS NOT NULL")

        elif operation_type in ["UPDATE", "DELETE", "SELECT"]:
            # 修改、删除、查询：全部字段参与 WHERE 条件
            # 主表字段 - 修复：正确处理字典格式
            for table_name, fields in field_requirements["primary_table_placeholder_fields"].items():
                for field in fields:
                    where_conditions.append(
                        f"{self.db_adapter.format_table_name(table_name)}.{self.db_adapter.format_field_name(field)} IS NOT NULL")

            # 关联表字段
            for table_name, fields in field_requirements["related_table_placeholder_fields"].items():
                for field in fields:
                    where_conditions.append(
                        f"{self.db_adapter.format_table_name(table_name)}.{self.db_adapter.format_field_name(field)} IS NOT NULL")

        return where_conditions

    def _build_template_table_connections(self, primary_table: str, field_requirements: Dict,
                                          used_relationships: List[Dict]) -> List[str]:
        """
        构建模板表之间的连接，只连接模板涉及的表
        """
        joins = []
        connected_tables = set([primary_table])

        # 获取模板涉及的所有表
        template_tables = set([primary_table])
        for table_name in field_requirements["related_table_placeholder_fields"].keys():
            template_tables.add(table_name)

        print(f"Template tables: {template_tables}")
        print(f"Available relationships: {len(used_relationships)}")

        # 只处理模板涉及的表之间的关联
        template_relationships = self._filter_relationships_for_template_tables(used_relationships, template_tables)

        print(f"Filtered relationships for template tables: {len(template_relationships)}")
        for rel in template_relationships:
            print(f"  {rel.get('from_table')}.{rel.get('from_column')} = {rel.get('to_table')}.{rel.get('to_column')}")

        # 策略1：尝试主表直接连接所有关联表
        direct_joins = self._try_direct_connections(primary_table, template_tables, template_relationships,
                                                    connected_tables)
        joins.extend(direct_joins)

        # 策略2：如果还有未连接的表，尝试通过已连接的表进行间接连接
        remaining_tables = template_tables - connected_tables
        if remaining_tables:
            print(f"Remaining unconnected tables: {remaining_tables}")
            indirect_joins = self._try_indirect_connections(connected_tables, remaining_tables, template_relationships)
            joins.extend(indirect_joins)

        print(f"Final joins: {joins}")
        return joins

    def _filter_relationships_for_template_tables(self, used_relationships: List[Dict], template_tables: set) -> List[
        Dict]:
        """
        过滤出只涉及模板表的关系
        """
        filtered_relationships = []

        print(f"Filtering relationships for template tables: {template_tables}")

        for relationship in used_relationships:
            from_table = relationship.get("from_table", "")
            to_table = relationship.get("to_table", "")

            # 只保留两个表都在模板范围内的关系
            if from_table in template_tables and to_table in template_tables:
                filtered_relationships.append(relationship)
                print(
                    f"Included: {from_table}.{relationship.get('from_column')} = {to_table}.{relationship.get('to_column')}")
            else:
                print(
                    f"Excluded: {from_table}.{relationship.get('from_column')} = {to_table}.{relationship.get('to_column')} (tables not in template)")

        return filtered_relationships

    def _try_direct_connections(self, primary_table: str, template_tables: set, template_relationships: List[Dict],
                                connected_tables: set) -> List[str]:
        """
        尝试主表直接连接模板涉及的表
        """
        joins = []

        for target_table in template_tables:
            if target_table == primary_table:
                continue

            # 查找主表到目标表的直接关系（只使用模板关系）
            relationship = self._find_direct_relationship(primary_table, target_table, template_relationships)

            if relationship:
                join_sql = self._build_join_from_relationship(relationship, primary_table, target_table)
                joins.append(join_sql)
                connected_tables.add(target_table)

        return joins

    def _try_indirect_connections(self, connected_tables: set, remaining_tables: set,
                                  template_relationships: List[Dict]) -> List[str]:
        """
        尝试通过已连接的表进行间接连接（只使用模板关系）
        """
        joins = []

        for target_table in remaining_tables:
            # 尝试通过已连接的表找到到目标表的路径（只使用模板关系）
            for connected_table in connected_tables:
                relationship = self._find_direct_relationship(connected_table, target_table, template_relationships)
                if relationship:
                    join_sql = self._build_join_from_relationship(relationship, connected_table, target_table)
                    joins.append(join_sql)
                    connected_tables.add(target_table)
                    break

        return joins

    def _find_direct_relationship(self, table1: str, table2: str, used_relationships: List[Dict]) -> Dict:
        """
        查找两个表之间的直接关系
        """
        for relationship in used_relationships:
            from_table = relationship.get("from_table", "")
            to_table = relationship.get("to_table", "")

            if (from_table == table1 and to_table == table2) or \
                    (from_table == table2 and to_table == table1):
                return relationship

        return None

    def _build_join_from_relationship(self, relationship: Dict, from_table: str, to_table: str) -> str:
        """
        根据关系构建 JOIN SQL
        """
        from_column = relationship.get("from_column", "")
        to_column = relationship.get("to_column", "")

        # 确定字段映射
        if relationship.get("from_table") == from_table:
            left_column = from_column
            right_column = to_column
        else:
            left_column = to_column
            right_column = from_column

        join_condition = f"{self.db_adapter.format_table_name(from_table)}.{self.db_adapter.format_field_name(left_column)} = {self.db_adapter.format_table_name(to_table)}.{self.db_adapter.format_field_name(right_column)}"

        return f"JOIN {self.db_adapter.format_table_name(to_table)} ON {join_condition}"

    def _execute_coordinated_data_sql(self, sql: str, field_requirements: Dict) -> Dict:
        """
        执行协调数据的大 SQL 查询，直接解析为 placeholder 格式
        """
        try:
            self.cursor.execute(sql)
            row = self.cursor.fetchone()

            if not row:
                return {}

            # 解析结果到 coordinated_data 格式
            coordinated_data = {}

            # 解析主表数据
            for table_name, fields in field_requirements["primary_table_placeholder_fields"].items():
                coordinated_data[table_name] = {}
                for field in fields:
                    # 从 SQL 结果中获取对应的值
                    value = self._get_field_value_from_result(field, row, self.cursor.description)
                    coordinated_data[table_name][field] = value

            # 解析关联表数据
            for table_name, fields in field_requirements["related_table_placeholder_fields"].items():
                coordinated_data[table_name] = {}
                for field in fields:
                    # 从 SQL 结果中获取对应的值
                    value = self._get_field_value_from_result(field, row, self.cursor.description)
                    coordinated_data[table_name][field] = value

            return coordinated_data

        except Exception as e:
            print(f"Error executing coordinated data SQL: {e}")
            return {}

    def _get_field_value_from_result(self, field_name: str, row: tuple, columns: tuple) -> any:
        """
        从 SQL 结果中获取指定字段的值
        """
        for i, col in enumerate(columns):
            if col[0] == field_name:  # 假设字段名不包含表前缀
                return self.db_adapter.handle_bytes_data(row[i])
            elif "." in col[0]:  # 处理带表前缀的字段名
                table_field = col[0].split(".", 1)
                if table_field[1] == field_name:
                    return self.db_adapter.handle_bytes_data(row[i])

        return None

    def _is_primary_table_fk_field(self, table_name: str, field_name: str, used_relationships: List[Dict]) -> bool:
        """
        判断字段是否为关联字段
        """
        for relationship in used_relationships:
            from_table = relationship.get("from_table", "")
            from_field = relationship.get("from_column", "")
            to_table = relationship.get("to_table", "")
            to_field = relationship.get("to_column", "")

            try:
                if (table_name == from_table and field_name == from_field) or \
                        (table_name == to_table and field_name == to_field):
                    return True
            except Exception as e:
                print(f"Error parsing relationship {relationship}, error: {e}")
                continue

        return False

    def _analyze_template_field_requirements(self, template: Dict, used_relationships: List[Dict]) -> Dict:
        """
        分析模板中的字段需求
        """
        placeholders = template.get("placeholders", [])
        primary_table = template.get("primary_table", "")

        requirements = {
            "primary_table_placeholder_fields": {},  # 改为字典格式
            "related_table_placeholder_fields": {},
            "primary_table_fk_fields": []
        }

        for placeholder in placeholders:
            table_name = placeholder.get("table", "")
            field_name = placeholder.get("field", "")

            if table_name == primary_table:
                if table_name not in requirements["primary_table_placeholder_fields"]:
                    requirements["primary_table_placeholder_fields"][table_name] = []
                requirements["primary_table_placeholder_fields"][table_name].append(field_name)

                if self._is_primary_table_fk_field(table_name, field_name, used_relationships):
                    requirements["primary_table_fk_fields"].append(field_name)
            else:
                # 关联表字段
                if table_name not in requirements["related_table_placeholder_fields"]:
                    requirements["related_table_placeholder_fields"][table_name] = []
                requirements["related_table_placeholder_fields"][table_name].append(field_name)

        return requirements

    def _get_coordinated_multi_table_data(self, primary_table: str, used_relationships: List[Dict],
                                          template: Dict) -> Dict:
        """
        使用大 SQL 直接获取协调的多表数据
        """
        try:
            # 1. 分析模板中的字段需求
            field_requirements = self._analyze_template_field_requirements(template, used_relationships)

            # 2. 获取操作类型
            operation_type = template.get("operation_type", "SELECT")

            # 3. 构建大 SQL 查询
            big_sql = self._build_coordinated_data_sql(primary_table, used_relationships, field_requirements,
                                                       operation_type)
            print(f"Generated SQL for coordinated data: {big_sql}")

            if not big_sql:
                print(f"Could not build SQL for coordinated data")
                return {}

            # 4. 执行大 SQL 查询
            coordinated_data = self._execute_coordinated_data_sql(big_sql, field_requirements)

            if not coordinated_data:
                print(f"Warning: Could not find suitable coordinated data for template, skipping")
                return {}

            return coordinated_data

        except Exception as e:
            print(f"Error getting coordinated multi-table data: {e}")
            return {}

    def enrich_templates(self, templates: List[Dict]) -> List[Dict]:

        """Enrich templates using verified relationships for data association."""
        print("Enriching templates with verified relationships...")

        enriched_templates = []

        for template in templates:
            zh_template_ = template.get("zh_template", "")
            primary_table_ = template.get("primary_table")
            related_tables_ = template.get("related_tables")
            placeholders_ = []
            for _ in template.get("placeholders", []):
                placeholders_.append(f" - {_.get('name')}")
            placeholders_ = "\n".join(placeholders_)
            used_relationships_ = []
            for _ in template.get("used_relationships", []):
                used_relationships_.append(
                    f" - from_table: {_.get('from_table')}, from_column: {_.get('from_column')}, to_table: {_.get('to_table')}, to_column: {_.get('to_column')}")
            used_relationships_ = "\n".join(used_relationships_)

            print("-" * 40)
            print(f"Template: {zh_template_}")
            print(f"Primary Table: {primary_table_}")
            print(f"Related Tables: {related_tables_}")
            print(f"Placeholders: \n{placeholders_}")
            print(f"Used Relationships: \n{used_relationships_}")

            template_type = template.get("template_type", "single_table")

            if template_type == "single_table":
                # 使用多变体生成
                print("\n--- Enrich single table template ---")
                enriched_template_list = self._enrich_single_table_template_multiple(template)
                enriched_templates.extend(enriched_template_list)
            elif template_type == "multi_table":
                # 使用多变体生成
                print("\n--- Enrich multi table template ---")
                enriched_template_list = self._enrich_multi_table_template_multiple(template)
                enriched_templates.extend(enriched_template_list)
            else:
                print(f"Unknown template type: {template_type}, skipping...")
                continue

        print(f"ENRICHMENT COMPLETED: {len(enriched_templates)} templates generated")
        return enriched_templates

    def _enrich_single_table_template_multiple(self, template: Dict) -> List[Dict]:
        """Enrich single table template with multiple variants."""
        variant_count = self.config.task_generation.variants_per_template
        primary_table = template.get("primary_table", "")
        operation_type = template.get("operation_type", "SELECT")

        if not primary_table:
            print("Error: No primary table specified")
            return []

        # Get primary table data
        table_data = self._get_table_data(primary_table, 1)
        if not table_data:
            print(f"Error: No data found for table {primary_table}")
            return []

        data = table_data[0]
        placeholder_values = {}
        placeholder_values_data_type = {}

        # Fill placeholders
        for placeholder_info in template.get("placeholders", []):
            placeholder_name = placeholder_info.get("name", "")
            table_name = placeholder_info.get("table", "")
            field_name = placeholder_info.get("field", "")
            field_type = placeholder_info.get("field_type", "")

            if table_name == primary_table and field_name in data:
                value = data[field_name]
                # Convert dots to hyphens in placeholder names
                formatted_placeholder_name = placeholder_name.replace(".", "-")
                placeholder_values[formatted_placeholder_name] = value
                placeholder_values_data_type[formatted_placeholder_name] = field_type
                print(f"Filled {formatted_placeholder_name} with {value}")

        # Use LLM to generate multiple variants
        adjusted_values_list = []
        if operation_type == "SELECT" or operation_type == "DELETE":
            adjusted_values_list.append(placeholder_values)
        else:
            adjusted_values_list = self._adjust_values_with_llm_multiple(template, placeholder_values,
                                                                         placeholder_values_data_type, variant_count)

        enriched_templates = []
        for i, adjusted_values in enumerate(adjusted_values_list):
            adjusted_values = self._clean_adjust_values(adjusted_values)
            # Fill template with adjusted values
            try:
                filled_template = template.get("template", "").format(**adjusted_values)
                filled_zh_template = template.get("zh_template", "").format(**adjusted_values)
                filled_sql = template.get("sql_template", "").format(**adjusted_values)
                filled_verification_sql = template.get("verification_sql", "").format(
                    **adjusted_values) if template.get(
                    "verification_sql") else ""
            except KeyError as e:
                print(f"Error: Missing placeholder value in variant {i + 1}: {e}")
                continue

            enriched_template = {
                "template": template.get("template", ""),
                "zh_template": template.get("zh_template", ""),
                "description": template.get("description", ""),
                "template_type": "single_table",
                "operation_type": operation_type,
                "primary_table": primary_table,
                "related_tables": template.get("related_tables", []),
                "sql_template": template.get("sql_template", ""),
                "verification_sql": template.get("verification_sql", ""),
                "used_relationships": template.get("used_relationships", []),
                "placeholders": template.get("placeholders", []),
                "filled_template": filled_template,
                "filled_zh_template": filled_zh_template,
                "filled_sql": filled_sql,
                "filled_verification_sql": filled_verification_sql,
                "placeholder_values": adjusted_values,  # Use adjusted values
                "placeholder_values_data_type": placeholder_values_data_type,
                "original_values": placeholder_values,  # Keep original values for comparison
                "data_source": data,
                "variant_id": i + 1  # Add variant ID
            }

            enriched_templates.append(enriched_template)
            print(f"Generated variant {i + 1}/{self.config.task_generation.variants_per_template + 1}")

        print(f"Completed: {len(enriched_templates)} variants generated")
        return enriched_templates

    def _clean_adjust_values(self, adjusted_values: Dict) -> Dict:
        cleaned_values = {}
        for key, value in adjusted_values.items():
            if isinstance(value, str):
                # Remove surrounding quotes
                cleaned_value = value.strip("'\"")
                cleaned_values[key] = cleaned_value
                print(f"Cleaned {key}: {value} -> {cleaned_value}")
            else:
                cleaned_values[key] = value
        return cleaned_values

    def _enrich_multi_table_template_multiple(self, template: Dict) -> List[Dict]:
        """Enrich multi table template with multiple variants."""
        variant_count = self.config.task_generation.variants_per_template
        primary_table = template.get("primary_table", "")
        related_tables = template.get("related_tables", [])
        used_relationships = template.get("used_relationships", [])
        operation_type = template.get("operation_type", "SELECT")

        if not primary_table:
            print("Error: No primary table specified")
            return []

        # Get coordinated multi-table data
        coordinated_data = self._get_coordinated_multi_table_data(
            primary_table, used_relationships, template
        )

        if not coordinated_data:
            print(f"Warning: Could not find fully coordinated data, returning primary table data only")
            return []
        print(f"Found coordinated data for {len(coordinated_data)} tables")
        print("Coordinated data:", coordinated_data)

        # Preprocess foreign key data, get all needed IDs
        placeholder_values, placeholder_values_data_type = self._preprocess_foreign_key_data(template, coordinated_data)

        # Use LLM to generate multiple variants
        adjusted_values_list = []
        if operation_type == "SELECT" or operation_type == "DELETE":
            adjusted_values_list.append(placeholder_values)
        else:
            adjusted_values_list = self._adjust_values_with_llm_multiple(template, placeholder_values,
                                                                         placeholder_values_data_type, variant_count)

        enriched_templates = []
        for i, adjusted_values in enumerate(adjusted_values_list):
            adjusted_values = self._clean_adjust_values(adjusted_values)
            # Fill template with adjusted values
            try:
                filled_template = template.get("template", "").format(**adjusted_values)
                filled_zh_template = template.get("zh_template", "").format(**adjusted_values)
                filled_sql = template.get("sql_template", "").format(**adjusted_values)
                filled_verification_sql = template.get("verification_sql", "").format(
                    **adjusted_values) if template.get(
                    "verification_sql") else ""
            except KeyError as e:
                print(f"Error: Missing placeholder value in variant {i + 1}: {e}")
                continue

            enriched_template = {
                "template": template.get("template", ""),
                "zh_template": template.get("zh_template", ""),
                "description": template.get("description", ""),
                "template_type": "multi_table",
                "operation_type": operation_type,
                "primary_table": primary_table,
                "related_tables": related_tables,
                "sql_template": template.get("sql_template", ""),
                "verification_sql": template.get("verification_sql", ""),
                "used_relationships": used_relationships,
                "placeholders": template.get("placeholders", []),
                "filled_template": filled_template,
                "filled_zh_template": filled_zh_template,
                "filled_sql": filled_sql,
                "filled_verification_sql": filled_verification_sql,
                "placeholder_values": adjusted_values,  # Use adjusted values
                "placeholder_values_data_type": placeholder_values_data_type,
                "original_values": placeholder_values,  # Keep original values for comparison
                "data_source": coordinated_data,
                "variant_id": i + 1  # Add variant ID
            }

            enriched_templates.append(enriched_template)
            print(f"Generated variant {i + 1}/{self.config.task_generation.variants_per_template + 1}")

        print(f"Completed: {len(enriched_templates)} variants generated")
        return enriched_templates

    def _enrich_multi_table_template_single(self, template: Dict) -> Dict:
        """Enrich multi table template with single variant (for association fields only)."""
        print(f"Enriching multi table template: {template.get('template', '')[:50]}...")

        primary_table = template.get("primary_table", "")
        related_tables = template.get("related_tables", [])
        used_relationships = template.get("used_relationships", [])

        if not primary_table:
            print("Error: No primary table specified")
            return None

        # Get coordinated multi-table data
        coordinated_data = self._get_coordinated_multi_table_data(
            primary_table, used_relationships
        )

        if not coordinated_data:
            print(f"Warning: Could not find fully coordinated data, returning primary table data only")
            # If coordinated data not found, use primary table data
            primary_data = self._get_table_data(primary_table, 1)
            if not primary_data:
                print(f"Error: No data found for primary table {primary_table}")
                return None
            coordinated_data = {primary_table: primary_data[0]}

        # Preprocess foreign key data, get all needed IDs
        placeholder_values, placeholder_values_data_type = self._preprocess_foreign_key_data(template, coordinated_data)

        # For association fields, use original values directly, don't call LLM
        adjusted_values = placeholder_values
        adjusted_values = self._clean_adjust_values(adjusted_values)

        # Fill template with adjusted values
        try:
            filled_template = template.get("template", "").format(**adjusted_values)
            filled_zh_template = template.get("zh_template", "").format(**adjusted_values)
            filled_sql = template.get("sql_template", "").format(**adjusted_values)
            filled_verification_sql = template.get("verification_sql", "").format(**adjusted_values) if template.get(
                "verification_sql") else ""
        except KeyError as e:
            print(f"Error: Missing placeholder value: {e}")
            return None

        return {
            "template": template.get("template", ""),
            "zh_template": template.get("zh_template", ""),
            "description": template.get("description", ""),
            "template_type": "multi_table",
            "operation_type": template.get("operation_type", "SELECT"),
            "primary_table": primary_table,
            "related_tables": related_tables,
            "sql_template": template.get("sql_template", ""),
            "verification_sql": template.get("verification_sql", ""),
            "used_relationships": used_relationships,
            "placeholders": template.get("placeholders", []),
            "filled_template": filled_template,
            "filled_zh_template": filled_zh_template,
            "filled_sql": filled_sql,
            "filled_verification_sql": filled_verification_sql,
            "placeholder_values": adjusted_values,  # Use adjusted values
            "placeholder_values_data_type": placeholder_values_data_type,
            "original_values": placeholder_values,  # Keep original values for comparison
            "data_source": coordinated_data,
            "variant_id": 1  # Single variant
        }

    def _get_table_data(self, table_name: str, limit: int = 1) -> List[Dict]:
        """Get data from a single table."""
        try:
            # Use adapter to format table name
            formatted_table = self.db_adapter.format_table_name(table_name)

            # Use different random ordering syntax based on database type
            order_clause = self.db_adapter.get_random_order_clause()

            self.cursor.execute(f"SELECT * FROM {formatted_table} {order_clause} LIMIT {limit}")
            rows = self.cursor.fetchall()

            if not rows:
                return []

            # Get column names
            columns_data = self.db_adapter.describe_table(table_name)
            columns = []
            for row in columns_data:
                columns.append(self.db_adapter.handle_bytes_data(row[0]))

            # Build data dictionary list
            table_data = []
            for row in rows:
                row_data = {}
                for i, col in enumerate(columns):
                    value = row[i]
                    row_data[col] = self.db_adapter.handle_bytes_data(value)
                table_data.append(row_data)

            return table_data

        except Exception as e:
            print(f"Error getting data from table {table_name}: {e}")
            # Ensure cleanup of any unprocessed result sets
            try:
                self.cursor.fetchall()
            except:
                pass
            return []

    def generate_answer_with_llm(self, sql_result, question_text, sql):
        """
        Use LLM to determine question type and generate appropriate eval answer
        """
        prompt = f"""
You are a database evaluation expert. Analyze the question and SQL result to generate an appropriate answer.

Question: {question_text}
SQL: {sql}
SQL Result: {sql_result}

Task: Generate an answer that can be used for evaluation. The answer should be a list of strings.

Question Type Analysis:
1. COUNT questions: "How many...", "Count...", "Number of..." → Return count as string
2. EXISTS questions: "Does...", "Is there...", "Exist..." → Return "Yes" or "No"
3. LIST questions: "List all...", "Show all...", "Find all..." → Return all result values
4. SINGLE questions: "What is...", "Get...", "Which..." → Return first value
5. MODIFICATION: INSERT/UPDATE/DELETE → Return affected rows count

CRITICAL RULES:
- For LIST questions that ask for "all" items, return ALL the actual values from SQL result
- For LIST questions without "all", return the count of results
- For COUNT queries, extract the count value from the result
- For EXISTS queries, return "Yes" if results exist, "No" if empty
- For SINGLE queries, return the first value from the result

Examples:
- "How many users are there?" → ["42"]
- "Does user John exist?" → ["Yes"]
- "List all projects" → ["Project A", "Project B", "Project C"]
- "List projects" → ["3"] (count)
- "What is the project name?" → ["Project Alpha"]
- "Update user status" → ["1"]

Return JSON:
{{
    "answer": ["value1", "value2", ...],
    "question_type": "count|exists|list|single|modification",
    "reasoning": "brief explanation of why this answer was chosen"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.openai.model,
                messages=[
                    {"role": "system", "content": "You are a database evaluation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            result = response.choices[0].message.content
            result = result.replace("```json", '').replace("```", '')
            result = json.loads(result)
            answer = result.get("answer", ["0"])
            reasoning = result.get("reasoning", "")

            print(f"LLM generated answer: {answer} (Reasoning: {reasoning})")
            return answer

        except Exception as e:
            print(f"Error generating answer with LLM: {e}")
            raise e

    def generate_questions_from_templates(self, enriched_templates: List[Dict]) -> List[Dict]:
        """Generate questions from templates that use verified relationships."""
        print("\n--- Generating questions from verified templates ---")

        questions = []

        for template in enriched_templates:
            template_ = template.get("template", "")
            sql_template = template.get("sql_template", "")
            filled_template = template.get("filled_template", "")
            filled_zh_template = template.get("filled_zh_template", "")
            filled_sql = template.get("filled_sql", "")
            filled_verification_sql = template.get("filled_verification_sql", "")
            placeholder_values = template.get("placeholder_values", {})
            placeholder_values_data_type = template.get("placeholder_values_data_type", {})
            template_type = template.get("template_type", "single_table")
            operation_type = template.get("operation_type", "SELECT")
            original_values = template.get("original_values", {})

            if not filled_template or not filled_sql:
                print(f"⚠️  Skipping template with missing filled_template or filled_sql")
                continue

            # Generate bilingual questions (English and Chinese)
            question_data = {
                "question": filled_template,
                "zh_question": filled_zh_template if filled_zh_template else "",
                "template": template_,
                "sql": filled_sql,
                "sql_template": sql_template,
                "verification_sql": filled_verification_sql if filled_verification_sql else "",
                "template_type": template_type,
                "operation_type": operation_type,
                "primary_table": template.get("primary_table", ""),
                "related_tables": template.get("related_tables", []),
                "used_relationships": template.get("used_relationships", []),
                "placeholders": template.get("placeholders", []),
                "placeholder_values": placeholder_values,
                "original_values": original_values,
                "placeholder_values_data_type": placeholder_values_data_type,
                "data_source": template.get("data_source", {})
            }

            questions.append(question_data)
            print(f"Success: Generated {template_type} {operation_type} bilingual question: {filled_template[:80]}...")
            if filled_zh_template:
                print(f"Chinese: {filled_zh_template[:80]}...")

        print(f"Generated {len(questions)} bilingual questions from templates")
        return questions

    def process_workflow_with_relationship_analysis(self, workflow: Dict) -> List[Dict]:
        """Complete workflow processing with relationship analysis."""
        print(f"\n" + "=" * 60)
        print(f"WORKFLOW PROCESSING: {workflow['name']}")
        print("=" * 60)

        try:
            verified_relationships = workflow.get("key_relationships")
            print(f"Difficulty: {self.config.task_generation.difficulty_level}")
            templates = self.generate_intent_templates(workflow, verified_relationships,
                                                       difficulty=self.config.task_generation.difficulty_level)
            enriched_templates = self.enrich_templates(templates)
            questions = self.generate_questions_from_templates(enriched_templates)
            validated_questions = self.execute_and_validate_questions(questions)

            print(f"WORKFLOW COMPLETED: {len(validated_questions)} validated questions")
            return validated_questions

        except Exception as e:
            print(f"Error: Error in workflow processing: {e}")
            return []

    def _adjust_values_with_llm_multiple(self, template: Dict, placeholder_values: Dict,
                                         placeholder_values_data_type: Dict,
                                         count: int = 3) -> List[Dict]:
        """Generate multiple variants while keeping all original constraints"""
        try:
            print("\nAdjusting values with LLM")

            operation_type = template.get("operation_type", "")

            # Get actual data for association fields (randomly select 5 for each type)
            available_data = self._get_available_data_for_association_fields(template)

            # Build prompt for multiple variants generation
            prompt = f"""
You are a database expert. Please adjust ONLY the VALUES, not the field names or SQL structure.

## Template Information
Template: {template.get('template', '')}
SQL Template: {template.get('sql_template', '')}
Operation Type: {operation_type}
SQL Database Type: {self.config.conf.database.type}
Critical: All values must remain valid SQL literals for the given SQL Database Type.

## Current Values (adjust only these)
**CRITICAL: Each value must strictly match its data type shown in parentheses**
{self._format_current_values_for_prompt(placeholder_values, placeholder_values_data_type)}

## Available Data for Association Fields (Choose from these existing values):
{self._format_available_data_for_prompt(available_data)}

## Adjustment Rules (CRITICAL - Must follow for ALL variants)
1. **ONLY adjust values, NEVER change field names**
2. **Association Fields**: Choose from the available data lists above
3. **Independent Fields**: Add random suffixes to avoid duplicates
4. **Empty Values**: Generate reasonable default values
5. **Required Fields**: Ensure all required fields have valid values

## Goal
Generate {count} different variants of the values. Each variant must follow ALL the rules above.

### Variant Diversity Requirements:
- **Variant 1**: Use realistic business-focused names and values
- **Variant 2**: Use technical/development-focused names and values  
- **Variant 3**: Use generic test-focused names and values
- **Each variant must be unique** but maintain data validity and consistency

### Important Constraints (Apply to ALL variants):
- Return ONLY the adjusted values in JSON format
- Keep the same key structure as current values
- Do not modify field names or SQL structure
- If a value is already good, keep it unchanged
- Association fields must choose from available data
- All variants must be valid for the operation type

Response format:
{{
    "variants": [
        {{
            "zt_user-realname": "John Smith",
            "zt_user-account": "johnsmith",
            "zt_project-budget": "50000"
        }},
        {{
            "zt_user-realname": "Jane Doe",
            "zt_user-account": "janedoe_dev",
            "zt_project-budget": "75000"
        }},
        {{
            "zt_user-realname": "Mike Johnson",
            "zt_user-account": "mikejohnson_test",
            "zt_project-budget": "30000"
        }}
    ]
}}
"""

            response = self.client.chat.completions.create(
                model=self.config.openai.model,
                messages=[
                    {"role": "system",
                     "content": "You are a database expert. Generate multiple unique variants while keeping all constraints."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # 增加随机性以确保变体多样性
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            variants = result.get("variants", [])

            # Validate all variants have required keys
            for i, variant in enumerate(variants):
                for key in placeholder_values.keys():
                    if key not in variant:
                        print(f"Warning: Missing key {key} in variant {i + 1}, using original value")
                        variant[key] = placeholder_values[key]

            print(f"Success: LLM generated {len(variants)} variants")
            return variants

        except json.JSONDecodeError as e:
            print(f"Error: LLM returned invalid JSON: {e}")
            print(f"LLM response: {result}")
            return []

        except Exception as e:
            print(f"Error: Error generating multiple variants: {e}")
            return []

    def _has_brace_placeholder(self, s: str) -> bool:
        return bool(re.search(r'\{[^}]+\}', s))


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== Hydra Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    print("=================================")

    output_base_dir = Path(cfg.output_base_dir)
    target_count = cfg.task_generation.target_count
    template_count = cfg.task_generation.template_count
    variants_per_template = cfg.task_generation.variants_per_template
    task_type = cfg.task_generation.task_type
    output_file = cfg.task_generation.output_file
    # verbose = cfg.task_generation.verbose
    if not cfg.task_generation.workflow_config:
        raise ValueError(
            "workflow_config is required. Please specify it via command line: platform.workflow_config=your_config.py")
    workflow_config = cfg.task_generation.workflow_config

    try:
        conn, cursor = DatabaseConnectionFactory.create_connection(cfg).connect()
        if not conn:
            print("Error: Database connection failed, exiting program")
            return
        print("Success: Database connection successful")

    except mysql.connector.Error as err:
        print(f"Error: Database connection failed: {err}")
        return

    try:
        if not cfg.openai.api_key:
            print("Error: Please set OPENAI_API_KEY environment variable")
            return

        client = OpenAI(
            api_key=cfg.openai.api_key,
            base_url=cfg.openai.base_url,
            timeout=cfg.openai.timeout
        )
        print("Success: OpenAI client created successfully")

    except Exception as err:
        print(f"Error: OpenAI client creation failed: {err}")
        return

    try:
        # Initialize enhanced generator
        generator = EnhancedTaskGenerator(client, cursor, cfg, custom_config_file=workflow_config)

        # Set task type and template count
        generator.task_type = task_type
        generator.template_count = template_count
        generator.variants_per_template = variants_per_template

        # Get configured workflows
        workflows = generator.get_configured_workflows()
        if not workflows:
            print("No workflows configured. Exiting.")
            return

        website_name = generator.website_name
        output_file_path = Path(output_file)
        time_str = datetime.now().strftime("%Y%m%d%H%M%S")
        if not output_file_path.is_absolute():
            output_dir = Path.joinpath(output_base_dir, "questions_bank", website_name)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = Path.joinpath(output_dir, f"{time_str}-{output_file_path.name}")

        # Start generation loop
        round_num = 1
        all_tasks = []
        while len(all_tasks) < target_count:
            print(f"\n--- Starting Enhanced Generation Round {round_num} ---")
            print(
                f"Goal: {target_count} | Current: {len(all_tasks)} | Needed: {target_count - len(all_tasks)}")

            # Select a random workflow for this round
            selected_workflow = random.choice(workflows)
            print(f"Selected workflow: {selected_workflow['name']}")

            # Use new relationship analysis workflow
            print(f"Processing workflow with relationship analysis: {selected_workflow['name']}")
            validated_questions = generator.process_workflow_with_relationship_analysis(selected_workflow)
            if not validated_questions:
                print("No questions generated. Trying next workflow...")
                continue

            # Add new tasks and save
            if validated_questions:
                # Append new tasks to JSONL file
                with open(output_file_path, 'a', encoding='utf-8') as f:
                    for task in validated_questions:
                        json.dump(task, f, ensure_ascii=False, default=str)
                        f.write('\n')

                all_tasks.extend(validated_questions)

                print(f"\n--- Round Summary ---")
                print(f"Generated {len(validated_questions)} new valid tasks in this round.")
                print(f"Progress: {len(all_tasks)} / {target_count} tasks.")
            else:
                print(f"\n--- Round Summary ---")
                print("No new valid tasks were generated in this round. Retrying...")

            round_num += 1

        print(f"\nTarget of {target_count} tasks reached. Final output saved to {output_file_path}.")
    except Exception as e:
        print(f"Error during task generation: {e}")
        print("Task generation failed, but cleanup will still be performed.")
    finally:
        if conn:
            db_adapter = DatabaseConnectionFactory.create_adapter(cursor, cfg)
            if db_adapter.is_connection_active(conn):
                cursor.close()
                conn.close()


if __name__ == "__main__":
    main()
