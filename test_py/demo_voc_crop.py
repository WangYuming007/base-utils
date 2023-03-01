# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 390737991@qq.com
    @Date   : 2022-11-29 18:11:47
    @Brief  :
"""
import os
import cv2
from tqdm import tqdm
from pybaseutils.dataloader import parser_voc
from pybaseutils import image_utils, file_utils


def save_object_crops(image, out_dir, bboxes, labels, image_id, extend=[], square=False, class_name=None, vis=False):
    image_id = image_id.split(".")[0]
    if square:
        bboxes = image_utils.get_square_bboxes(boxes=bboxes)
    if extend:
        bboxes = image_utils.extend_xyxy(bboxes, scale=extend, valid_range=[])
    if vis:
        image = image_utils.draw_image_bboxes_labels(image, bboxes, labels, class_name=class_name,
                                                     thickness=2, fontScale=0.8, drawType="custom")
        image_utils.cv_show_image("image", image)
    crops = image_utils.get_bboxes_crop(image, bboxes)
    for i, img in enumerate(crops):
        if class_name:
            name = class_name[int(labels[i])]
        else:
            name = labels[i]
        file = os.path.join(out_dir, name, "{}_{}_{:0=3d}.jpg".format(image_id, name, i))
        if vis:
            print(file)
            image_utils.cv_show_image("crop", img)
        file_utils.create_file_path(file)
        cv2.imwrite(file, img)


if __name__ == "__main__":
    """
    """
    # data_root = "/home/dm/nasdata/dataset-dmai/handwriting/word-det/word-v3"
    data_root = "/home/dm/nasdata/dataset/csdn/Eyeglasses/dataset/video/others/images11"
    # data_root = "/home/dm/nasdata/dataset/csdn/Eyeglasses/dataset/eyeglasses-v1/face"
    # data_root = "/home/dm/nasdata/dataset/csdn/Eyeglasses/dataset/eyeglasses-v2/face-eyeglasses"
    # data_root = "/home/dm/nasdata/dataset/csdn/Eyeglasses/dataset/eyeglasses-v2/face"
    # data_root = "/home/dm/nasdata/dataset/csdn/Eyeglasses/dataset/eyeglasses-test/face-eyeglasses"
    # data_root = "/home/dm/nasdata/dataset/csdn/Eyeglasses/dataset/eyeglasses-test/face"
    # out_dir = os.path.join(os.path.dirname(image_dir), "crops")
    out_dir = os.path.join(data_root, "crops")
    dataset = parser_voc.VOCDataset(filename=None,
                                    data_root=data_root,
                                    anno_dir=None,
                                    image_dir=None,
                                    class_name=None,
                                    transform=None,
                                    check=False,
                                    use_rgb=False,
                                    shuffle=False)
    print("have num:{}".format(len(dataset)))
    class_name = dataset.class_name
    extend = [1.1, 1.1]
    for i in tqdm(range(len(dataset))):
        data = dataset.__getitem__(i)
        image, targets, image_id = data["image"], data["target"], data["image_id"]
        bboxes, labels = targets[:, 0:4], targets[:, 4:5]
        save_object_crops(image, out_dir, bboxes, labels, image_id, extend=extend, class_name=class_name, vis=False)
