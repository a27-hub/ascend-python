import yaml

files = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
for i in range(1, 22):
    print(i)
    with open(f'{i}.yaml', 'r', encoding='utf8') as f:
        data = yaml.safe_load(f)
    for file in data:
        with open(file['filename'], 'w') as f:
            f.write(file['data'])