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
import sys
input_avi_dir = 'datasets/MSVD/raw_videos/'
input_img_dir = 'datasets/MSVD/8frames/'
output_img_dir = 'datasets/MSVD/opticalflow_8frames/'
txtname = sys.argv[1]

with open("0.txt", "w") as f0, \
        open("1.txt", "w") as f1, \
        open("2.txt", "w") as f2, \
open("3.txt", "w") as f3, \
open("4.txt", "w") as f4, \
open("5.txt", "w") as f5, \
open("6.txt", "w") as f6, \
open("7.txt", "w") as f7, \
open("8.txt", "w") as f8, \
open("9.txt", "w") as f9, \
open("10.txt", "w") as f10, \
open("11.txt", "w") as f11, \
open("12.txt", "w") as f12, \
open("13.txt", "w") as f13, \
open("14.txt", "w") as f14, \
open("15.txt", "w") as f15, \
open("16.txt", "w") as f16, \
open("17.txt", "w") as f17, \
open("18.txt", "w") as f18, \
open("19.txt", "w") as f19:
    for i, filename in enumerate(os.listdir(input_avi_dir)):
        print(i)
        print(filename)
        if i < 900:
            continue
        if i % 20 == 0:
            f0.write(filename.split('.avi')[0])
            f0.write('\n')
        elif i  % 20 == 1:
            f1.write(filename.split('.avi')[0])
            f1.write('\n')
        elif i  % 20 == 2:
            f2.write(filename.split('.avi')[0])
            f2.write('\n')
        elif i  % 20 == 3:
            f3.write(filename.split('.avi')[0])
            f3.write('\n')
        elif i  % 30 == 4:
            f4.write(filename.split('.avi')[0])
            f4.write('\n')
        elif i  % 20 == 5:
            f5.write(filename.split('.avi')[0])
            f5.write('\n')
        elif i  % 20 == 6:
            f6.write(filename.split('.avi')[0])
            f6.write('\n')
        elif i  % 20 == 7:
            f7.write(filename.split('.avi')[0])
            f7.write('\n')
        elif i  % 20 == 8:
            f8.write(filename.split('.avi')[0])
            f8.write('\n')
        elif i  % 20 == 9:
            f9.write(filename.split('.avi')[0])
            f9.write('\n')

        elif i % 20 == 10:
            f10.write(filename.split('.avi')[0])
            f10.write('\n')
        elif i  % 20 == 11:
            f11.write(filename.split('.avi')[0])
            f11.write('\n')
        elif i  % 20 == 12:
            f12.write(filename.split('.avi')[0])
            f12.write('\n')
        elif i  % 20 == 13:
            f13.write(filename.split('.avi')[0])
            f13.write('\n')
        elif i  % 20 == 14:
            f14.write(filename.split('.avi')[0])
            f14.write('\n')
        elif i  % 20 == 15:
            f15.write(filename.split('.avi')[0])
            f15.write('\n')
        elif i  % 20 == 16:
            f16.write(filename.split('.avi')[0])
            f16.write('\n')
        elif i  % 20 == 17 or i  % 20 == 18 or i  % 20 == 19:
            f17.write(filename.split('.avi')[0])
            f17.write('\n')
        '''
        elif i  % 20 == 18 or i  % 20 == 19:
            f18.write(filename.split('.avi')[0])
            f18.write('\n')
        '''




for filename in open(txtname):
    print(filename)
    filename = filename.split('\n')[0]
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
