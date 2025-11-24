from pathlib import Path

# Using pathlib (recommended)
total_size = sum(f.stat().st_size for f in Path('./temp/test_cache').rglob('*') if f.is_file())
print(f"Total size: {total_size:,} bytes")

# Convert to human-readable format
def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

print(f"Total size: {format_bytes(total_size)}")