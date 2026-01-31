import os
import datetime

# Configuration
POSTS_DIR = "_posts"
RETENTION_DAYS = 30

def cleanup_old_posts():
    if not os.path.exists(POSTS_DIR):
        return

    now = datetime.datetime.now()
    count = 0

    for filename in os.listdir(POSTS_DIR):
        if filename.endswith(".md"):
            try:
                # Jekyll filenames start with YYYY-MM-DD
                file_date_str = "-".join(filename.split("-")[:3])
                file_date = datetime.datetime.strptime(file_date_str, "%Y-%m-%d")
                
                age = (now - file_date).days
                if age > RETENTION_DAYS:
                    os.remove(os.path.join(POSTS_DIR, filename))
                    print(f"🗑️ Deleted archive: {filename}")
                    count += 1
            except Exception as e:
                print(f"⚠️ Skipping {filename}: {e}")

    print(f"✨ Cleanup complete. Removed {count} old posts.")

if __name__ == "__main__":
    cleanup_old_posts()
