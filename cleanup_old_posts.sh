#!/usr/bin/env bash
# Run this from the ROOT of your ai-news-agent repo (where _posts/ lives).
# It finds and removes:
#   1) Posts whose title leaked an LLM refusal ("none found", "no specific topic", etc.)
#   2) Posts that are exact/near-duplicate titles of an earlier post (keeps the earliest)
#   3) The header image referenced by each removed post (if present)
#
# It stages the deletions with `git rm` and creates ONE commit.
#
# Usage:
#   ./cleanup_old_posts.sh          Interactive (asks for y/N confirmation)
#   ./cleanup_old_posts.sh --yes    Non-interactive (for CI / autonomous runs)

set -euo pipefail

POSTS_DIR="_posts"
IMG_DIR="assets/img"
AUTO_YES=false

for arg in "$@"; do
  case "$arg" in
  --yes | -y)
    AUTO_YES=true
    ;;
  esac
done

if [ ! -d "$POSTS_DIR" ]; then
  echo "❌ Can't find $POSTS_DIR — run this script from the repo root."
  exit 1
fi

declare -A seen_titles
to_delete=()

shopt -s nullglob
for f in "$POSTS_DIR"/*.md; do
  title=$(grep -m1 '^title:' "$f" | sed -E 's/^title:[[:space:]]*"(.*)"[[:space:]]*$/\1/')
  lower=$(echo "$title" | tr '[:upper:]' '[:lower:]')

  # 1) Junk / refusal titles that leaked from a failed LLM generation
  if echo "$lower" | grep -qiE "none found|no specific topic|i could not find|i cannot find|the text only mentions|not enough information"; then
    echo "🗑️  Junk title detected: $f"
    echo "     \"$title\""
    to_delete+=("$f")
    continue
  fi

  # 2) Exact duplicate title of an earlier post
  if [[ -n "${seen_titles[$lower]:-}" ]]; then
    echo "🗑️  Duplicate of \"${seen_titles[$lower]}\": $f"
    to_delete+=("$f")
  else
    seen_titles[$lower]="$title"
  fi
done

if [ ${#to_delete[@]} -eq 0 ]; then
  echo "✅ No junk or duplicate posts found. Nothing to do."
  exit 0
fi

echo
echo "About to delete ${#to_delete[@]} file(s):"
printf '  %s\n' "${to_delete[@]}"
echo

if ! $AUTO_YES; then
  read -r -p "Proceed with deletion + commit? [y/N] " confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted. No changes made."
    exit 1
  fi
fi

for f in "${to_delete[@]}"; do
  # Try to find and remove the header image this post referenced
  img_rel=$(grep -A1 '^image:' "$f" 2>/dev/null | grep 'path:' | sed -E 's/^[[:space:]]*path:[[:space:]]*\/?(.*)[[:space:]]*$/\1/' || true)
  if [[ -n "${img_rel:-}" ]]; then
    if [ -f "$img_rel" ]; then
      echo "🗑️  Removing orphaned image: $img_rel"
      git rm -f "$img_rel"
    fi
  fi
  git rm -f "$f"
done

if $AUTO_YES; then
  # In CI this runs as part of a larger commit staged by the workflow, so
  # just leave the deletions staged -- don't commit here.
  echo "✅ Done (non-interactive). ${#to_delete[@]} file(s) staged for removal."
else
  git commit -m "🧹 Clean up duplicate/junk auto-generated posts"
  echo
  echo "✅ Done. Review the commit (git show --stat HEAD), then:"
  echo "     git push origin main"
fi
