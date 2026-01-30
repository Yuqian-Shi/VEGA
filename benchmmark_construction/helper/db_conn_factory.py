# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Authors :       sundapeng.sdp
   Date：          2025/8/20
   Description :
-------------------------------------------------
"""
__author__ = 'sundapeng.sdp'


class DatabaseAdapter:
    """Database adapter base class"""

    def __init__(self, cursor):
        self.cursor = cursor

    def describe_table(self, table_name):
        """Get table structure"""
        raise NotImplementedError

    def get_row_count(self, table_name):
        """Get table row count"""
        raise NotImplementedError

    def get_foreign_keys(self, table_name):
        """Get foreign key information"""
        raise NotImplementedError

    def is_connection_active(self, conn):
        """Check if connection is active"""
        raise NotImplementedError

    def handle_bytes_data(self, data):
        """Handle byte type data"""
        raise NotImplementedError

    def format_table_name(self, table_name):
        """Format table name (handle quotes etc.)"""
        raise NotImplementedError

    def format_field_name(self, field_name):
        """Format field name"""
        raise NotImplementedError

    def get_all_table_names(self):
        """Get all table names in database"""
        raise NotImplementedError

    def get_random_order_clause(self):
        """Get random order clause"""
        raise NotImplementedError


class MySQLAdapter(DatabaseAdapter):
    """MySQL database adapter"""

    def describe_table(self, table_name):
        """Get table structure"""
        self.cursor.execute(f"DESCRIBE `{table_name}`")
        return self.cursor.fetchall()

    def get_row_count(self, table_name):
        """Get table row count"""
        self.cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
        result = self.cursor.fetchone()
        return int(result[0]) if result and result[0] is not None else 0

    def get_foreign_keys(self, table_name):
        """Get foreign key information"""
        self.cursor.execute(f"""
            SELECT 
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
            WHERE TABLE_SCHEMA = DATABASE() 
            AND TABLE_NAME = %s
            AND REFERENCED_TABLE_NAME IS NOT NULL
        """, (table_name,))
        return self.cursor.fetchall()

    def is_connection_active(self, conn):
        """Check if connection is active"""
        return conn.is_connected()

    def handle_bytes_data(self, data):
        """Handle byte type data"""
        if isinstance(data, bytes):
            return data.decode('utf-8')
        return str(data) if data is not None else None

    def format_table_name(self, table_name):
        """Format table name"""
        return f"`{table_name}`"

    def format_field_name(self, field_name):
        """Format field name"""
        return f"`{field_name}`"

    def get_all_table_names(self):
        """Get all table names in database"""
        self.cursor.execute("SHOW TABLES")
        tables = self.cursor.fetchall()
        table_names = []
        for row in tables:
            table_name = self.handle_bytes_data(row[0])
            if table_name:
                table_names.append(table_name)
        return table_names

    def get_random_order_clause(self):
        return "ORDER BY RAND()"


class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL database adapter"""

    def describe_table(self, table_name):
        """Get table structure"""
        self.cursor.execute(f"""
            SELECT 
                column_name as field,
                data_type as type,
                is_nullable as null,
                CASE 
                    WHEN column_default IS NOT NULL THEN 'DEFAULT'
                    WHEN is_nullable = 'NO' THEN 'NOT NULL'
                    ELSE ''
                END as key,
                column_default as default_value,
                '' as extra
            FROM information_schema.columns 
            WHERE table_name = %s 
            ORDER BY ordinal_position
        """, (table_name,))
        return self.cursor.fetchall()

    def get_row_count(self, table_name):
        """Get table row count"""
        self.cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        result = self.cursor.fetchone()
        return int(result[0]) if result and result[0] is not None else 0

    def get_foreign_keys(self, table_name):
        """Get foreign key information"""
        self.cursor.execute(f"""
            SELECT 
                kcu.column_name,
                ccu.table_name as referenced_table_name,
                ccu.column_name as referenced_column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' 
            AND tc.table_name = %s
        """, (table_name,))
        return self.cursor.fetchall()

    def is_connection_active(self, conn):
        """Check if connection is active"""
        try:
            self.cursor.execute("SELECT 1")
            return True
        except:
            return False

    def handle_bytes_data(self, data):
        """Handle byte type data"""
        return str(data) if data is not None else None

    def format_table_name(self, table_name):
        """Format table name"""
        return f'"{table_name}"'

    def format_field_name(self, field_name):
        """Format field name"""
        return f'"{field_name}"'

    def get_all_table_names(self):
        """Get all table names in database"""
        self.cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        tables = self.cursor.fetchall()
        table_names = []
        for row in tables:
            table_name = self.handle_bytes_data(row[0])
            if table_name:
                table_names.append(table_name)
        return table_names

    def get_random_order_clause(self):
        return "ORDER BY RANDOM()"


class MySQLConnection:
    def __init__(self, cfg):
        self.cfg = cfg

    def connect(self):
        try:
            import mysql.connector
            db_config = {
                "host": self.cfg.conf.database.host,
                "port": self.cfg.conf.database.port,
                "user": self.cfg.conf.database.user,
                "password": self.cfg.conf.database.password,
                "database": self.cfg.conf.database.database,
                "charset": self.cfg.conf.database.charset,
                "autocommit": self.cfg.conf.database.autocommit,
                "connect_timeout": self.cfg.conf.database.connect_timeout,
                "read_timeout": self.cfg.conf.database.read_timeout,
                "write_timeout": self.cfg.conf.database.write_timeout
            }

            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            print("MySQL database connection successful")
            return conn, cursor

        except Exception as err:
            print(f"MySQL database connection failed: {err}")
            return None, None


class PostgreSQLConnection:
    def __init__(self, cfg):
        self.cfg = cfg

    def connect(self):
        try:
            import psycopg2
            db_config = {
                "host": self.cfg.conf.database.host,
                "port": self.cfg.conf.database.port,
                "user": self.cfg.conf.database.user,
                "password": self.cfg.conf.database.password,
                "database": self.cfg.conf.database.database,
                "connect_timeout": self.cfg.conf.database.connect_timeout
            }

            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            print("PostgreSQL database connection successful")
            return conn, cursor

        except Exception as err:
            print(f"PostgreSQL database connection failed: {err}")
            return None, None


class DatabaseConnectionFactory:
    """Database connection factory"""

    @staticmethod
    def create_connection(cfg):
        """Create database connection based on configuration"""
        db_type = cfg.conf.database.type.lower()

        if db_type == "mysql":
            return MySQLConnection(cfg)
        elif db_type == "postgresql":
            return PostgreSQLConnection(cfg)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    @staticmethod
    def create_adapter(cursor, cfg):
        """Create database adapter"""
        db_type = cfg.conf.database.type.lower()

        if db_type == "mysql":
            return MySQLAdapter(cursor)
        elif db_type == "postgresql":
            return PostgreSQLAdapter(cursor)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
