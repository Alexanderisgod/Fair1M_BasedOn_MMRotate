import os
from pathlib import Path

'''过滤出来161张图片
'''
'''
annoted_file_names = os.listdir('/root/FAIR1M_plane_1_1024_512/test/labelTxt')

with open("/root/mmrotate/tools/test_filenames.txt", 'r') as file:
    data = file.read()
    file_names = data.split('\n')[:-1]

for x in file_names:
    if x not in annoted_file_names:
        print(x)
'''

'''将 FAIR1M_plane_1_1024_512/test/labelTxt   中的标注信息进行合并
'''


lable_txts_dir=Path('/root/FAIR1M_plane_1_1024_512/test/labelTxt/')
lable_txts_path = [lable_txts_dir/x for x in os.listdir(lable_txts_dir)]

with open("/root/mmrotate/tools/dota/annos.txt") as file:
    for label_txt_path in lable_txts_path:
        # with open()
        pass