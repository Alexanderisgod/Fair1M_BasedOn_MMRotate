import os


annoted_file_names = os.listdir('/root/FAIR1M_plane_1_1024_512/test/labelTxt')

with open("/root/mmrotate/tools/test_filenames.txt", 'r') as file:
    data = file.read()
    file_names = data.split('\n')[:-1]

for x in file_names:
    if x not in annoted_file_names:
        print(x)

