import json

with open('data/output.txt', 'r', encoding='utf-8') as f_in, \
     open('data/output.jsonl', 'w', encoding='utf-8') as f_out:
    
    lines = [line.strip() for line in f_in.readlines() if line.strip()]  # 去除空行
    
    i = 0
    while i < len(lines):
        if lines[i].startswith('错误形式：'):
            wrong = lines[i][5:]  # 去掉 "错误形式："
            if i+1 < len(lines) and lines[i+1].startswith('正确形式：'):
                correct = lines[i+1][5:]  # 去掉 "正确形式："
                json_obj = {'wrong': wrong, 'correct': correct}
                f_out.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
            i += 2
        else:
            i += 1