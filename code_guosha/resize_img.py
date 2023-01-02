'''
if is_train == True:
    self.raw_video_crop_list = [
        Resize(self.img_res),
        RandomCrop((self.img_res, self.img_res)),
        ClipToTensor(channel_nb=3),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
else:
    self.raw_video_crop_list = [
        Resize(self.img_res),
        CenterCrop((self.img_res, self.img_res)),
        ClipToTensor(channel_nb=3),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
self.raw_video_prcoess = Compose(self.raw_video_crop_list)
'''
import os
input_avi_dir = 'datasets/MSVD/raw_videos/'
input_img_dir = 'datasets/MSVD/8frames/'
output_img_dir = 'datasets/MSVD/opticalflow_8frames/'

with open("0.txt", "w") as f0, \
        open("1.txt", "w") as f1, \
        open("2.txt", "w") as f2, \
open("3.txt", "w") as f3, \
open("4.txt", "w") as f4, \
open("5.txt", "w") as f5, \
open("6.txt", "w") as f6, \
open("7.txt", "w") as f7, \
open("8.txt", "w") as f8, \
open("9.txt", "w") as f9:
    for i, filename in enumerate(os.listdir(input_avi_dir)):

        if i % 10 == 0:
            f0.write(filename.split('.avi')[0])
        elif i  % 10 == 1:
            f1.write(filename.split('.avi')[0])
        elif i  % 10 == 2:
            f2.write(filename.split('.avi')[0])
        elif i  % 10 == 3:
            f3.write(filename.split('.avi')[0])
        elif i  % 10 == 4:
            f4.write(filename.split('.avi')[0])
        elif i  % 10 == 5:
            f5.write(filename.split('.avi')[0])
        elif i  % 10 == 6:
            f6.write(filename.split('.avi')[0])
        elif i  % 10 == 7:
            f7.write(filename.split('.avi')[0])
        elif i  % 10 == 8:
            f8.write(filename.split('.avi')[0])
        elif i  % 10 == 9:
            f9.write(filename.split('.avi')[0])

for filename in os.listdir(input_avi_dir):
    print(filename)
    filename = filename.split('.avi')[0]
    for itr in range(1, 9):
        startidx = 0
        endidx = 0
        scale = '1.0'
        if itr <= 7:
            startidx = itr
            endidx = itr + 1
            scale = '1.0'
        else:
            startidx = itr
            endidx = itr - 1
            scale = '-1.0'

        leftpng = input_img_dir + filename + '_frame' + str(startidx).zfill(4) + '.jpg'
        rightpng = input_img_dir + filename + '_frame' + str(endidx).zfill(4) + '.jpg'
        outputpng = output_img_dir + filename + '_frame' + str(startidx).zfill(4) + '.flo'
        cmd = 'python3 run.py --model default --one ' + leftpng + ' --two ' + rightpng + ' --out ' + outputpng + ' --scale ' + scale
        print(cmd)
        os.system(cmd)