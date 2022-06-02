import mmcv
import math
import numpy as np
import glob

def coordinate_convert_r(box):
    w, h = box[2:-1]
    theta = -box[-1]
    x_lu, y_lu = -w/2, h/2
    x_ru, y_ru = w/2, h/2
    x_ld, y_ld = -w/2, -h/2
    x_rd, y_rd = w/2, -h/2

    x_lu_ = math.cos(theta)*x_lu + math.sin(theta)*y_lu + box[0]
    y_lu_ = -math.sin(theta)*x_lu + math.cos(theta)*y_lu + box[1]

    x_ru_ = math.cos(theta) * x_ru + math.sin(theta) * y_ru + box[0]
    y_ru_ = -math.sin(theta) * x_ru + math.cos(theta) * y_ru + box[1]

    x_ld_ = math.cos(theta) * x_ld + math.sin(theta) * y_ld + box[0]
    y_ld_ = -math.sin(theta) * x_ld + math.cos(theta) * y_ld + box[1]

    x_rd_ = math.cos(theta) * x_rd + math.sin(theta) * y_rd + box[0]
    y_rd_ = -math.sin(theta) * x_rd + math.cos(theta) * y_rd + box[1]

    convert_box = np.asarray([x_lu_, y_lu_, x_ru_, y_ru_, x_rd_, y_rd_, x_ld_, y_ld_]).reshape(4, -1)
    # convert_box = np.int0(convert_box).clip(1, 1024)
    convert_box = np.int0(convert_box)
    return convert_box


def get_file_order(ann_folder):
    ann_names = []
    ann_files = glob.glob(ann_folder + '/*.png')
    for ann_file in ann_files:
        ann_names.append(ann_file.split('/')[-1][:-4]+'.txt')
    # 文件名加入文件
    with open("/root/mmrotate/tools/dota/test_filenames.txt", 'w') as file:
        for ann_name in ann_names:
            print(ann_name)
            file.write(ann_name+'\n')


def merge_annos(src_dir, dst_file):
    '''将 FAIR1M_plane_1_1024_512/test/labelTxt   中的标注信息进行合并
    '''
    lable_txts_dir=Path('/root/FAIR1M_plane_1_1024_512/test/labelTxt/')
    lable_txts_path = [lable_txts_dir/x for x in os.listdir(lable_txts_dir)]

    with open("/root/mmrotate/tools/dota/detection_gt_annos.txt", 'w') as file:
        for label_txt_path in lable_txts_path:
            with open(label_txt_path, 'r') as each_file:
                data = each_file.readlines()
                file_name = osp.split(label_txt_path)[-1][:-4]
                for cords in data:
                    file.write("{} {}".format(file_name, cords))



OUT_PKL = "/root/mmrotate/out.pkl"                          # model output pkl
IMAGESETFILE = "/root/mmrotate/tools/dota/imagesetfile.txt" # test images' name
DETPATH = "/root/mmrotate/tools/dota/Task1_Airplane.txt"    # path to save convert bboxs points

if __name__=="__main__":
    with open(IMAGESETFILE, 'r') as file:
        file_names = file.readlines()
    
    result = mmcv.load(OUT_PKL)
    thresh = 0.3
    count = 0
    with open(DETPATH, 'w') as file:
        for index, (boxes, file_name) in enumerate(zip(result,file_names)):
            for box in boxes[0]:
                if box[-1]<thresh:
                    continue
                count+=1
                # convert to 4 points
                rect=coordinate_convert_r(box[:-1])
                cords=" ".join([str(x) for x in rect.reshape(-1)])
                file.write(file_name[:-1]+' '+str(box[-1])+' '+cords+'\n')
                # print("{}, {}, {}, {} ".format(rect[0],rect[1], rect[2], rect[3]))

    print("the predicted bboxs' count is :{}".format(count))
