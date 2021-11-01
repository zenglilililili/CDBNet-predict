import torch
import numpy as np
import torch.nn as nn
from imageio import imsave


def getSliceTrain(ct_path):
    train_img = np.load(ct_path)
    train_img = (train_img - np.min(train_img)) / 255

    train_img = train_img.reshape((1, 1, train_img.shape[0], train_img.shape[1]))

    return train_img


def tensor_to_np(tensor):
    imgtmp = tensor.cpu().numpy()
    return imgtmp


def classifier(model1_path, model2_path, device, ct_path, save_path):
    train_img = getSliceTrain(ct_path)

    with torch.no_grad():
        # class Model
        classModel = torch.load(model1_path)
        classModel.eval()
        classModel = classModel.to(device)

        inputs = torch.FloatTensor(train_img)
        inputs = inputs.to(device)
        output = classModel(inputs)

        slice_img_predict = tensor_to_np(output)

        if np.sum(slice_img_predict) > 65:
            # segModel
            segModel = torch.load(model2_path)
            segModel.eval()
            segModel = segModel.to(device)

            segInputs = torch.FloatTensor(train_img)
            segInputs = segInputs.to(device)
            segOutput, _ = segModel(segInputs)

            segSlice_img_predict = tensor_to_np(segOutput)

            segSlice_img_predict = segSlice_img_predict.reshape(512, 512)
            # print(segSlice_img_predict.shape)

            imsave(save_path, segSlice_img_predict)
        else:
            img = np.zeros((512, 512))
            imsave(save_path, img)


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


pos = "CTV"  # 预测部位，便于以后修改扩展其他部位
result_path = "bestModel/"
testSlice = 'model1.pkl'  # 分类模型
SegSlice = 'model2.pkl'  # 分割模型

evaluation = DiceEva()
gpu = '0'
device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")

# 输入的病例的切片编号
str = "2190,2191,2192,2193,2194,2195,2196,2197,2198,2199,2200,2201,2202,2203,2204,2205,2206,2207,2208,2209,2210,2211,2212,2213,2214,2215,2216,2217,2218,2219,2220,2221,2222,2223,2224,2225,2226,2227,2228,2229,2230,2231,2232,2233,2234,2235,2236,2237,2238,2239,2240,2241,2242,2243,2244,2245,2246,2247,2248,2249,2250,2251,2252,2253,2254,2255,2256,2257,2258,2259,2260,2261,2262,2263,2264,2265,2266,2267,2268,2269,2270,2271,2272,2273,2274,2275,2276,2277,2278,2279,2280,2281,2282,2283,2284,2285,2286,2287,2288,2289,2290,2291,2292,2293,2294,2295,2296,2297,2298,2299,2300,2301,2302,2303,2304,2305,2306,2307,2308,2309,2310,2311,2312,2313,2314,2315,2316,2317,2318,2319,2320,2321,2322,2323,2324,2325,2326,2327,2328,2329,2330,2331,2332,2333,2334,2335,2336,2337,2338,2339,2340,2341,2342,2343,2344,2345,2346,2347,2348,2349,2350,2351,2352,3640,3641,3642,3643,3644,3645,3646,3647,3648,3649,3650,3651,3652,3653,3654,3655,3656,3657,3658,3659,3660,3661,3662,3663,3664,3665,3666,3667,3668,3669,3670,3671,3672,3673,3674,3675,3676,3677,3678,3679,3680,3681,3682,3683,3684,3685,3686,3687,3688,3689,3690,3691,3692,3693,3694,3695,3696,3697,3698,3699,3700,3701,3702,3703,3704,3705,3706,3707,3708,3709,3710,3711,3712,3713,3714,3715,3716,3717,3718,3719,3720,3721,3722,3723,3724,3725,3726,3727,3728,3729,3730,3731,3732,3733,3734,3735,3736,3737,3738,3739,3740,3741,3742,3743,3744,3745,3746,3747,3748,3749,3750,3751,3752,3753,3754,3755,3756,3757,3758,3759,3760,3761,3762,3763,3764,3765,3766,3767,3768,3769,3770,3771,3772,3773,3774,3775,3776,3777,3778,3779,3780,3781,3782,3783,3784,3785,3786,3787,3788,3789,3790,3791,3792,3793,3794,3795,3796,3797,3798,3799,3800,3801,3802,3803,3804,3805,3806,3807,3808,3809,3810,3811"
slice_List = str.split(',')

for i in slice_List:
    save_path = "data/predict/" + i + ".jpg"
    ct_path = "data/ct/" + i + ".npy"

    classifier(result_path + testSlice, result_path + SegSlice, device, ct_path, save_path)
