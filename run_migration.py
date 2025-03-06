import os
import sys
import subprocess

# Set the RENDER environment variable to indicate we're in production
os.environ['RENDER'] = 'true'

# Run the migration script
try:
    print("Starting database migration...")
    subprocess.run([sys.executable, 'migrate_db.py'], check=True)
    print("Migration completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Migration failed with error code {e.returncode}")
    sys.exit(1)

