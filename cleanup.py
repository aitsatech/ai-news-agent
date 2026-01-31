import os
import sys
from datetime import datetime, timedelta

def clean_old_posts(posts_dir="_posts", days_to_keep=30):
    # Check if directory exists
    if not os.path.exists(posts_dir):
        print(f"Directory {posts_dir} not found. Skipping cleanup.")
        return # This is now safely inside a function

    now = datetime.now()
    cutoff_date = now - timedelta(days=days_to_keep)
    
    print(f"Cleaning posts older than: {cutoff_date.strftime('%Y-%m-%d')}")

    count = 0
    for filename in os.listdir(posts_dir):
        if filename.endswith(".md"):
            try:
                # Jekyll format: YYYY-MM-DD-title.md
                date_str = filename[:10]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

                if file_date < cutoff_date:
                    file_path = os.path.join(posts_dir, filename)
                    os.remove(file_path)
                    print(f"🗑️ Deleted: {filename}")
                    count += 1
            except ValueError:
                # Skip files that don't start with a valid date
                continue

    print(f"✨ Cleanup finished. Removed {count} posts.")

if __name__ == "__main__":
    clean_old_posts()
