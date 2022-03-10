import mmcv

data_path = '/mnt/cache/chenzehui/code/DDOD/data/crowdhuman/annotations/val.json'
data = mmcv.load(data_path)

print(data['annotations'][0])
print(data['images'][0])
print(data['categories'])