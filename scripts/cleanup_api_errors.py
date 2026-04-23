import json

input_file = "/home/liuqian/DR/Tongyi/output/alibaba/tongyi-deepresearch-30b-a3b/bc-zn10-single-turn-redundant-skip/iter1.jsonl"
error_keyword = "openai_api_key"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned_lines = [line for line in lines if error_keyword not in line]

with open(input_file, "w", encoding="utf-8") as f:
    for line in cleaned_lines:
        f.write(line)

print(f"原始记录数: {len(lines)}")
print(f"清除后记录数: {len(cleaned_lines)}")
print(f"删除记录数: {len(lines) - len(cleaned_lines)}")
