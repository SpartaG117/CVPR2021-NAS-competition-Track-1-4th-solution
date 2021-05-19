import json
import os
import sys

output_dir = sys.argv[1]
files = os.listdir(output_dir)

d = dict()
for file in files:
    if file.endswith('.json'):
        with open(os.path.join(output_dir, file), 'r') as f:
            d.update(json.load(f))


with open(os.path.join(output_dir, 'submission_final.json'), 'w') as f:
    json.dump(d, f)