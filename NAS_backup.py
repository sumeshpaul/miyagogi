import os
import shutil
from datetime import datetime

# Define source and destination
SOURCE_FILE = "/mnt/ssd2tb/projects/miyagogi/logs.csv"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
DEST_DIR = "/mnt/nas_llm/backups"
DEST_FILE = f"{DEST_DIR}/miyagogi_logs_prod_v1_{TIMESTAMP}.csv"

# Ensure destination exists
os.makedirs(DEST_DIR, exist_ok=True)

# Copy and verify
shutil.copy2(SOURCE_FILE, DEST_FILE)
if os.path.exists(DEST_FILE) and os.path.getsize(DEST_FILE) > 0:
    print(f"✅ Backup successful: {DEST_FILE}")
else:
    print("❌ Backup failed.")
