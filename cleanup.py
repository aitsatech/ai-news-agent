import os
import re
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


def _referenced_images(posts_dir="_posts"):
    """
    Scan whatever posts are still present and collect the basenames of every
    image referenced in front matter (image: / path: ...). Any image on this
    list must NEVER be deleted by clean_old_assets, no matter its age --
    this is what prevents "post survives, header image gets pruned" bugs.
    """
    referenced = set()
    if not os.path.exists(posts_dir):
        return referenced

    for filename in os.listdir(posts_dir):
        if not filename.endswith(".md"):
            continue
        file_path = os.path.join(posts_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read(2000)  # front matter is always near the top
        except Exception:
            continue

        match = re.search(r'^\s*path:\s*/?(.+?)\s*$', text, flags=re.MULTILINE)
        if match:
            referenced.add(os.path.basename(match.group(1).strip()))

    return referenced


def clean_old_assets(img_dir="assets/img", posts_dir="_posts", days_to_keep=7):
    """
    Deletes header images older than the specified days, but ONLY if they
    are not referenced by any post still on disk (see _referenced_images).
    This keeps a short image-retention window for genuinely orphaned images
    without breaking header images on posts that are still live -- e.g. a
    post kept for 30 days must never lose its image at day 7.
    """
    if not os.path.exists(img_dir):
        print(f"📁 Directory {img_dir} not found. Skipping asset cleanup.")
        return

    print(f"🖼️ Cleaning unreferenced images older than {days_to_keep} days...")

    now = time.time()
    cutoff_seconds = now - (days_to_keep * 86400)

    # Files to NEVER delete
    protected_files = ["avatar.png", "aitsa.png", "favicon.ico"]

    still_referenced = _referenced_images(posts_dir)
    if still_referenced:
        print(f"🔗 {len(still_referenced)} image(s) still referenced by live posts -- protected.")

    count = 0
    for filename in os.listdir(img_dir):
        if filename in protected_files:
            continue
        if filename in still_referenced:
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
    # 1. Clean the text posts first (keep 30 days)
    clean_old_posts(days_to_keep=30)

    print("-" * 30)

    # 2. Clean unreferenced images (keep 7 days) -- run AFTER post cleanup
    #    so _referenced_images() reflects the posts that actually survived.
    clean_old_assets(days_to_keep=7)
