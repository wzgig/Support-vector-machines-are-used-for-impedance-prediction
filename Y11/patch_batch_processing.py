
import os

target_file = r"e:\ruanjian\GitHubDesktop\Support-vector-machines-are-used-for-impedance-prediction\Y11\batch_processing.py"

with open(target_file, "r", encoding="utf-8") as f:
    content = f.read()

# Replace OUTPUT_FILE usage
old_output = 'fname = OUTPUT_FILE.replace(".csv", f"_{elem}.csv")'
new_output = 'fname = output_file.replace(".csv", f"_{elem}.csv")'

# Replace if __name__ block
old_main = 'if __name__ == "__main__":\n    # Windows 下使用 multiprocess 必须放在 if __name__ == "__main__": 下\n    run_batch()'
new_main = 'if __name__ == "__main__":\n    # Windows 下使用 multiprocess 必须放在 if __name__ == "__main__": 下\n    print("=== 开始批量处理所有数据集 ===")\n    for ds in DATASETS:\n        run_batch(ds)\n    print("\\n=== 所有任务完成 ===")'

# Try to find the old main block more robustly
if old_main not in content:
    # Try with single quotes or different line endings handled by python's universal newlines
    pass 

# Since read() handles newlines, let's just replace carefully.
# We know the last few lines.

lines = content.splitlines()

# Modify OUTPUT_FILE line
for i, line in enumerate(lines):
    if 'fname = OUTPUT_FILE.replace' in line:
        lines[i] = lines[i].replace('OUTPUT_FILE', 'output_file')
        print("Replaced OUTPUT_FILE at line", i+1)

# Modify main block
# It's at the end.
if lines[-1].strip() == 'run_batch()':
    lines.pop() # remove run_batch()
    lines.append('    print("=== 开始批量处理所有数据集 ===")')
    lines.append('    for ds in DATASETS:')
    lines.append('        run_batch(ds)')
    lines.append('    print("\\n=== 所有任务完成 ===")')
    print("Replaced main block")

new_content = "\n".join(lines)

with open(target_file, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Done.")
