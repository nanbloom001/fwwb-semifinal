import json
import sys
import os

def convert(json_path, md_path):
    with open(json_path, 'r') as f:
        content = json.load(f)
    # Remove '本页总览' prefix
    if content.startswith('本页总览'):
        content = content[4:]
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w') as f:
        f.write(content)
    print(f"OK: {md_path} ({len(content)} chars)")
    os.remove(json_path)

if __name__ == '__main__':
    json_path = sys.argv[1]
    md_path = sys.argv[2]
    convert(json_path, md_path)
