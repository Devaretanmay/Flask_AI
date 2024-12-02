from sqlalchemy import create_engine, inspect
import os
from dotenv import load_dotenv

load_dotenv()

# Create database connection
engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# Get inspector
inspector = inspect(engine)

# Print schema for all tables
for table_name in inspector.get_table_names():
    print(f"\nTable: {table_name}")
    print("Columns:")
    for column in inspector.get_columns(table_name):
        print(f"- {column['name']} ({column['type']})")
    
    print("\nForeign Keys:")
    for fk in inspector.get_foreign_keys(table_name):
        print(f"- {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}") 