import os
import time
from datetime import datetime, timedelta

def clean_old_posts(posts_dir="_posts", days_to_keep=30):
    """Deletes Markdown posts older than the specified days."""
    if not os.path.exists(posts_dir):
        print(f"📁 Directory {posts_dir} not found. Skipping post cleanup.")
        return

    now = datetime.now()
    cutoff_date = now - timedelta(days=days_to_keep)
    print(f"📅 Cleaning posts older than: {cutoff_date.strftime('%Y-%m-%d')}")

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
                    print(f"🗑️ Deleted Post: {filename}")
                    count += 1
            except ValueError:
                # Skip files that don't match the Jekyll date format
                continue
    print(f"✨ Post cleanup finished. Removed {count} posts.")

def clean_old_assets(img_dir="assets/img", days_to_keep=7):
    """Deletes header images older than the specified days, preserving branding."""
    if not os.path.exists(img_dir):
        print(f"📁 Directory {img_dir} not found. Skipping asset cleanup.")
        return

    print(f"🖼️ Cleaning images older than {days_to_keep} days...")
    
    now = time.time()
    cutoff_seconds = now - (days_to_keep * 86400)
    
    # Files to NEVER delete
    protected_files = ["avatar.png", "aitsa.png", "favicon.ico"]
    
    count = 0
    for filename in os.listdir(img_dir):
        if filename in protected_files:
            continue
            
        file_path = os.path.join(img_dir, filename)
        
        if os.path.isfile(file_path):
            file_time = os.path.getmtime(file_path)
            if file_time < cutoff_seconds:
                try:
                    os.remove(file_path)
                    print(f"🗑️ Deleted Image: {filename}")
                    count += 1
                except Exception as e:
                    print(f"⚠️ Error deleting {filename}: {e}")
                    
    print(f"✨ Asset cleanup finished. Removed {count} images.")

if __name__ == "__main__":
    # 1. Clean the text posts (Keep 30 days)
    clean_old_posts(days_to_keep=30)
    
    print("-" * 30)
    
    # 2. Clean the heavy images (Keep 7 days)
    clean_old_assets(days_to_keep=7)
