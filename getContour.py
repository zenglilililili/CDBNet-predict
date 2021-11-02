import torch
import numpy as np
import time
import random
import argparse
import csv
import os
from tqdm import tqdm
import shutil
import pandas as pd
import torch.nn as nn
import torch
from hausdorff import hausdorff_distance
import cv2
from matplotlib import pyplot as plt


def getdataset(path, k):
    print("path = " + path + "/k_fold_twoStage_DB3/" + str(k) + "/testset.txt")
    with open(path + "/k_fold_twoStage_DB3/" + str(k) + "/testset.txt", "r") as f:  # 打开文件
        trainset = f.read()  # 读取文件
    trainset = trainset.replace("\n", "").split(',')
    newTrainset = []
    for item in trainset:
        if int(item) <= 24845 and int(item) >= 24651:
            continue
        newTrainset.append(item)
    print("test_list length = " + str(len(trainset)))
    print("new test_list length = " + str(len(newTrainset)))
    # newTrainset = newTrainset[0:50]
    return newTrainset


def getSliceTrain(root_path, slice_index):
    train_img_name = root_path + '/bmp/CT_CTV/' + str(slice_index) + ".npy"
    train_img = np.load(train_img_name)
    # train_img = getWin(train_img, -128, 256)
    train_img = (train_img - np.min(train_img)) / 255

    ctv = 0
    try:
        train_label_name = root_path + '/bmp/Label_CTV/' + str(slice_index) + ".npy"
        train_label = np.load(train_label_name)
        train_label = np.where(train_label > 0, 1, 0)
        ctv = 1
    except:
        train_label = np.zeros((512, 512))

    train_img = train_img.reshape((1, 1, train_img.shape[0], train_img.shape[1]))
    train_label = train_label.reshape((1, 1, train_label.shape[0], train_label.shape[1]))

    return train_img, train_label, ctv


class DiceEva(nn.Module):
    def __init__(self):
        super(DiceEva, self).__init__()

    def forward(self, output, target):
        smooth = 0.00001
        if torch.is_tensor(output):
            output = output.cpu().numpy()
        if torch.is_tensor(target):
            target = target.data.cpu().numpy()
        target = target[0, 0, :, :]
        output = np.where(output > 0.1, 1, 0)
        output_flat = output.flatten()
        target_flat = target.flatten()

        intersection = output_flat * target_flat

        Dice = (2.0 * intersection.sum() + smooth) / (output_flat.sum() + target_flat.sum() + smooth)
        return Dice


def tensor_to_np(tensor):
    imgtmp = tensor.cpu().numpy()
    # imgtmp = np.where(imgtmp > 0.02, 1, 0)
    return imgtmp


def getContour(map, img, color):
    kernel = np.ones((5, 5), dtype=np.uint8)
    true = cv2.dilate(img, kernel, 1)  # 1:迭代次数，也就是执行几次膨胀操作

    gray = cv2.cvtColor(true, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(map, contours, -1, color, 1)

    return map


def getLabel(slice_count):
    # img = np.where(img > 0.02, 255, 0)
    # cv2.imwrite("ct.bmp", img)
    # cv2.imwrite("label.bmp", img_bmpCT)

    preUnet = cv2.imread("img/preUnet/" + str(slice_count) + ".bmp")
    preDB3 = cv2.imread("img/preDB3/" + str(slice_count) + ".bmp")
    true = cv2.imread("img/true/" + str(slice_count) + ".bmp")
    ct = cv2.imread("img/CT/" + str(slice_count) + ".bmp")

    # plt.subplot(141)
    # plt.imshow(ct, cmap='gray')
    # plt.subplot(142)
    # plt.imshow(true, cmap='gray')

    # kernel = np.ones((25, 25), dtype=np.uint8)
    # true = cv2.dilate(true, kernel, 1)  # 1:迭代次数，也就是执行几次膨胀操作

    # imgt = np.zeros((512, 512))
    # gray = cv2.cvtColor(true, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(imgt, contours, -1, (255, 255, 255), 1)
    # cv2.drawContours(ct, contours, -1, color, 1)

    map = getContour(ct, true, (0, 0, 255))
    map = getContour(map, preUnet, (0, 255, 0))
    map = getContour(map, preDB3, (0, 255, 255))

    # plt.subplot(143)
    # plt.imshow(imgt, cmap='gray')
    # plt.subplot(144)
    # plt.imshow(img_bmpCT, cmap='gray')
    # plt.show()
    return map


def classifier(root_path, slice_count, device, evaluation, model_path1, model_path2):
    train_img, train_label, ctv = getSliceTrain(root_path, slice_count)

    with torch.no_grad():
        # class Model
        # print(model_path1)
        classModel = torch.load(model_path1)
        classModel.eval()
        classModel = classModel.to(device)

        inputs = torch.FloatTensor(train_img)
        inputs = inputs.to(device)
        output = classModel(inputs)

        slice_img_predict = tensor_to_np(output)
        loss_i = evaluation(slice_img_predict, train_label)

        if np.sum(slice_img_predict) > 65 and ctv == 1:

            # segModel
            segModel = torch.load(model_path2)
            segModel.eval()
            segModel = segModel.to(device)

            segInputs = torch.FloatTensor(train_img)
            segInputs = segInputs.to(device)
            segOutput, _ = segModel(segInputs)

            segSlice_img_predict = tensor_to_np(segOutput)
            # if np.sum(segSlice_img_predict) > 65:
            #     pre = 1

            loss_i2 = evaluation(segSlice_img_predict, train_label)

            train_img_name = root_path + '/bmp/CT_CTV/' + str(slice_count) + ".npy"
            img_CT = np.load(train_img_name)

            plt.subplot(121)
            plt.imshow(img_CT.reshape(512, 512), cmap='gray')
            # plt.show()

            train_label = np.where(train_label > 0.2, 255, 0)
            slice_img_predict = np.where(slice_img_predict > 0.2, 255, 0)
            segSlice_img_predict = np.where(segSlice_img_predict > 0.2, 255, 0)

            cv2.imwrite("img/CT/" + str(slice_count) + ".bmp", img_CT)
            cv2.imwrite("img/preUnet/" + str(slice_count) + ".bmp", slice_img_predict[0, 0, :, :])
            cv2.imwrite("img/preDB3/" + str(slice_count) + ".bmp", segSlice_img_predict[0, 0, :, :])
            cv2.imwrite("img/true/" + str(slice_count) + ".bmp", train_label[0, 0, :, :])

            try:
                img_CT = getLabel(slice_count)  # 红色
            except:
                print("Error: model err")
                return
                # print("Error: model1 err")

            # plt.subplot(122)
            # plt.imshow(img_CT, cmap='gray')
            # plt.show()

            cv2.imwrite('resPic/' + str(round(loss_i2 - loss_i, 2)) + "_" + str(slice_count) + ".bmp", img_CT)


def main():
    path = '/home/ahazeng/桌面/dataset/loadData/'
    root_path = "/home/ahazeng/桌面/dataset/loadData/"
    model_path = "./bestModel/"
    k = 2
    testset = getdataset(path, k)
    gpu = '0'
    device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    print(device)
    evaluation = DiceEva()

    for slice_count in tqdm(testset):
        # print(slice_count)
        classifier(root_path, slice_count, device, evaluation, model_path + "Unet/model.pkl",
                   model_path + "DB3/model.pkl")
    # cv2.imwrite('resPic/' + str(slice_count) + "_" + str(dice)+".bmp", pic)


main()