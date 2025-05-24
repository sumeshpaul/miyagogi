import os
import shutil
from datetime import datetime

# Source and destination paths
SOURCE_DIR = "/mnt/ssd2tb/projects/miyagogi"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
DEST_DIR = f"/mnt/nas_llm/backups/miyagogi_project_backup_{TIMESTAMP}"

# Perform recursive folder copy
shutil.copytree(SOURCE_DIR, DEST_DIR)

# Confirm backup
if os.path.exists(DEST_DIR):
    print(f"✅ Full project backup successful:\n→ {DEST_DIR}")
else:
    print("❌ Backup failed.")
