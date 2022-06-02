from mmrotate.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = '/root/mmrotate/configs/kfiou/r3det_kfiou_ln_r50_fpn_1x_dota_oc.py'
checkpoint_file = '/root/mmrotate/work_dirs/r3det_kfiou_ln_r50_fpn_1x_dota_oc/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/root/FAIR1M_plane_1_1024_512/test/images/7__1__1024___0.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
# model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)