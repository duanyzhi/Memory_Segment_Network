import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import scipy.io as scio

def NNS_HMM():
    '''
    |--------|--------|--------|--------|--------|
    |        |        |        |        |        |

    N = 4, sigma=2, threshold = 4, NNS = 20, raw baysian_information:
        acc:0.841  acc_nns:0.859  acc_hmm:0.895 acc_nns_hmm:0.922
    N = 2, sigma=2, threshold = 4, NNS = 20:
        acc:0.841  acc_nns:0.859  acc_hmm:0.886 acc_nns_hmm:0.895
    N = 8, sigma=2, threshold = 4, NNS = 20:
        acc:0.841  acc_nns:0.859  acc_hmm:0.872 acc_nns_hmm:0.888
    N = 16, sigma=2, threshold = 4, NNS = 20:
        acc:0.841  acc_nns:0.859  acc_hmm:0.771 acc_nns_hmm:0.804

    # pre_bay with fix bay info is better:
    N = 2, sigma=2, threshold = 4, NNS = 20:
        acc:0.850  acc_nns:0.872  acc_hmm:0.904 acc_nns_hmm:0.904
    N = 4, sigma=2, threshold = 4, NNS = 20:
        acc:0.850  acc_nns:0.872  acc_hmm:0.909 acc_nns_hmm:0.959
    N = 8, sigma=2, threshold = 4, NNS = 20:
        acc:0.850  acc_nns:0.872  acc_hmm:0.936 acc_nns_hmm:0.941
    N = 12, sigma=2, threshold = 4, NNS = 20:
        acc:0.850  acc_nns:0.872  acc_hmm:0.941 acc_nns_hmm:0.913
    N = 16, sigma=2, threshold = 4, NNS = 20:
        acc:0.850  acc_nns:0.872  acc_hmm:0.918 acc_nns_hmm:0.908

    # sigma 0.5 1 2 5, N = 4, threshold = 4, NNS = 20
    sigma = 0.5: acc_nns_hmm:0.904  sigma = 1: acc_nns_hmm:0.954
    sigma = 2: acc_nns_hmm:0.959    sigma = 5: acc_nns_hmm:0.959
    sigma = 10: acc_nns_hmm:0.949

    # threshold change, sigma = 0.5, N = 4, NNS = 20
    threshold = 1:  acc_hmm:0.703 acc_nns_hmm:0.562
    threshold = 2:  acc_hmm:0.785 acc_nns_hmm:0.753
    threshold = 3:  acc_hmm:0.845 acc_nns_hmm:0.863
    threshold = 4:  acc_hmm:0.868 acc_nns_hmm:0.904
    threshold = 5:  acc_hmm:0.881 acc_nns_hmm:0.927
    threshold = 6:  acc_hmm:0.890 acc_nns_hmm:0.941

    # threshold change, sigma = 1, N = 4, NNS = 20
    threshold = 1:  acc_hmm:0.744 acc_nns_hmm:0.721
    threshold = 2:  acc_hmm:0.817 acc_nns_hmm:0.836
    threshold = 3:  acc_hmm:0.868 acc_nns_hmm:0.927
    threshold = 4:  acc_hmm:0.878 acc_nns_hmm:0.954
    threshold = 5:  acc_hmm:0.886 acc_nns_hmm:0.968
    threshold = 6:  acc_hmm:0.895 acc_nns_hmm:0.973

    # threshold change, sigma = 2, N = 4, NNS = 20
    threshold = 1:  acc_hmm:0.749 acc_nns_hmm:0.785
    threshold = 2:  acc_hmm:0.831 acc_nns_hmm:0.872
    threshold = 3:  acc_hmm:0.890 acc_nns_hmm:0.927
    threshold = 4:  acc_hmm:0.909 acc_nns_hmm:0.959
    threshold = 5:  acc_hmm:0.913 acc_nns_hmm:0.968
    threshold = 6:  acc_hmm:0.922 acc_nns_hmm:0.973

    # threshold change, sigma = 5, N = 4, NNS = 20
    threshold = 1:  acc_hmm:0.813 acc_nns_hmm:0.845
    threshold = 2:  acc_hmm:0.858 acc_nns_hmm:0.899
    threshold = 3:  acc_hmm:0.899 acc_nns_hmm:0.936
    threshold = 4:  acc_hmm:0.918 acc_nns_hmm:0.959
    threshold = 5:  acc_hmm:0.927 acc_nns_hmm:0.968
    threshold = 6:  acc_hmm:0.936 acc_nns_hmm:0.973

    # threshold change, sigma = 10, N = 4, NNS = 20
    threshold = 1:  acc_hmm:0.836 acc_nns_hmm:0.868
    threshold = 2:  acc_hmm:0.872 acc_nns_hmm:0.918
    threshold = 3:  acc_hmm:0.890 acc_nns_hmm:0.941
    threshold = 4:  acc_hmm:0.895 acc_nns_hmm:0.950
    threshold = 5:  acc_hmm:0.909 acc_nns_hmm:0.963
    threshold = 6:  acc_hmm:0.913 acc_nns_hmm:0.968

    # loc_threshold change, N = 4, sigma = 2, NNS = 20
    loc_threshold=1:  0ｍ
        acc:0.790  acc_nns:0.801  acc_hmm:0.749 acc_nns_hmm:0.785
    loc_threshold=2:  10m
        acc:0.818  acc_nns:0.836  acc_hmm:0.831 acc_nns_hmm:0.872
    loc_threshold=3:  20m
        acc:0.841  acc_nns:0.859  acc_hmm:0.890 acc_nns_hmm:0.927
    loc_threshold=4:  30m
        acc:0.850  acc_nns:0.872  acc_hmm:0.909 acc_nns_hmm:0.959
    loc_threshold=5:  40m
        acc:0.863  acc_nns:0.886  acc_hmm:0.913 acc_nns_hmm:0.968
    loc_threshold=6:  50m
        acc:0.873  acc_nns:0.895  acc_hmm:0.922 acc_nns_hmm:0.973

    # N change, loc_threshold = 1, sigma = 2, NNS = 20
    N = 2: acc_hmm:0.799 acc_nns_hmm:0.790
    N = 4: acc_hmm:0.749 acc_nns_hmm:0.785
    N = 8: acc_hmm:0.735 acc_nns_hmm:0.744
    N = 12: acc_hmm:0.758 acc_nns_hmm:0.753
    N = 16: acc_hmm:0.753 acc_nns_hmm:0.758

    # N change, loc_threshold = 2, sigma = 2, NNS = 20
    N = 2: acc_hmm:0.845 acc_nns_hmm:0.845
    N = 4: acc_hmm:0.831 acc_nns_hmm:0.872
    N = 8: acc_hmm:0.831 acc_nns_hmm:0.836
    N = 12: acc_hmm:0.849 acc_nns_hmm:0.817
    N = 16: acc_hmm:0.836 acc_nns_hmm:0.826

    # N change, loc_threshold = 3, sigma = 2, NNS = 20
    N = 2: acc_hmm:0.886 acc_nns_hmm:0.890
    N = 4: acc_hmm:0.890 acc_nns_hmm:0.927
    N = 8: acc_hmm:0.899 acc_nns_hmm:0.886
    N = 12: acc_hmm:0.913 acc_nns_hmm:0.863
    N = 16: acc_hmm:0.890 acc_nns_hmm:0.877

    # N change, loc_threshold = 4, sigma = 2, NNS = 20
    N = 2: acc_hmm:0.904 acc_nns_hmm:0.904
    N = 4: acc_hmm:0.909 acc_nns_hmm:0.959
    N = 8: acc_hmm:0.936 acc_nns_hmm:0.941
    N = 12: acc_hmm:0.940 acc_nns_hmm:0.913
    N = 16: acc_hmm:0.890 acc_nns_hmm:0.877

    # N change, loc_threshold = 5, sigma = 2, NNS = 20
    N = 2: acc_hmm:0.913 acc_nns_hmm:0.913
    N = 4: acc_hmm:0.913 acc_nns_hmm:0.968
    N = 8: acc_hmm:0.949 acc_nns_hmm:0.959
    N = 12: acc_hmm:0.959 acc_nns_hmm:0.936
    N = 16: acc_hmm:0.941 acc_nns_hmm:0.931

    # N change, loc_threshold = 6, sigma = 2, NNS = 20
    N = 2: acc_hmm:0.922 acc_nns_hmm:0.922
    N = 4: acc_hmm:0.922 acc_nns_hmm:0.973
    N = 8: acc_hmm:0.954 acc_nns_hmm:0.963
    N = 12: acc_hmm:0.973 acc_nns_hmm:0.954
    N = 16: acc_hmm:0.954 acc_nns_hmm:0.950
    '''
    error_threshold = [0, 10, 20, 30, 40, 50]  # meter
    # sigma change with error threshold
    acc_sigma_05 = [0.562, 0.753, 0.863, 0.904, 0.927, 0.941]
    acc_sigma_1 = [0.721, 0.836, 0.927, 0.954, 0.968, 0.973]
    acc_sigma_2 = [0.785, 0.872, 0.927, 0.959, 0.968, 0.973]
    acc_sigma_5 = [0.845, 0.899, 0.936, 0.959, 0.968, 0.973]
    acc_sigma_10 = [0.868, 0.918, 0.941, 0.950, 0.963, 0.968]

def plot_N_change():
    '''
    N改变
    '''
    # save error threshold and
    error_threshold = [0, 10, 20, 30, 40, 50]  # meter

    nns_N_2 = [0.790, 0.845, 0.890, 0.904, 0.913, 0.922] # N=2 different loc_threshold
    nns_N_4 = [0.785, 0.872, 0.927, 0.959, 0.968, 0.973]
    nns_N_8 = [0.744, 0.836, 0.886, 0.941, 0.959, 0.963]
    nns_N_12 = [0.753, 0.817, 0.863, 0.913, 0.936, 0.954]
    nns_N_16 = [0.758, 0.826, 0.877, 0.877, 0.931, 0.950]

    # 设置风格
    # plt.style.use('fivethirtyeight')
    # plt.style.use('ggplot')

    # 设置xtick坐标轴刻度线的方向，in,out,inout
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'

    fig, ax = plt.subplots()

    # this sets the figure on which we'll draw all the subplots
    # plt.figure(figsize=(10,8))

    # 设置横纵坐标轴范围
    ax.set_ylim(0.75, 1)
    ax.set_xlim(0, 50)

    # 加图像标题
    # plt.title("MSN")

    # 加横纵坐标
    plt.xlabel("Error Threshold(m)")
    plt.ylabel("Location Accuarcy in the given Error Threshold(%)")

    # 图中加横纵线分割
    # plt.grid(True)

    # 颜色配色网站colordrop.io
    plt.plot(error_threshold, nns_N_2, color="#00a03e", marker='+', linewidth=1,  label='MSMN,N=2', markersize=8)
    plt.plot(error_threshold, nns_N_4, color="#f2317f", marker='*',  linewidth=1, label='MSMN,N=4', markersize=8)
    plt.plot(error_threshold, nns_N_8, color="#3399FF", marker='s', linewidth=1,  label='MSMN,N=8', markersize=8)
    plt.plot(error_threshold, nns_N_12, color="#040000", marker='d',  linewidth=1, label='MSMN,N=12', markersize=8)
    # plt.plot(error_threshold, nns_N_16, color="#ea7070", marker='^',  linewidth=2, label='NNS-HMMP,N=16', markersize=8)

    # 加图注释
    # plt.legend(loc="best")
    plt.legend(loc='upper left', frameon=True)

    # Space plots a bit
    plt.subplots_adjust(hspace=0.25, wspace=0.40)

    plt.show()

def plot_MP_HMMP():
    # save error threshold and
    error_threshold = [0, 10, 20, 30, 40, 50]  # meter

    MP = [0.790, 0.818, 0.841, 0.850, 0.863, 0.873] # N=2 different loc_threshold
    HMMP = [0.749, 0.831, 0.890, 0.909, 0.913, 0.922]
    NNS_HMMP = [0.785, 0.872, 0.927, 0.959, 0.968, 0.973]


    # 设置风格
    # plt.style.use('fivethirtyeight')
    # plt.style.use('ggplot')

    # 设置xtick坐标轴刻度线的方向，in,out,inout
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'

    fig, ax = plt.subplots()

    # this sets the figure on which we'll draw all the subplots
    # plt.figure(figsize=(10,8))

    # 设置横纵坐标轴范围
    ax.set_ylim(0.75, 1)
    ax.set_xlim(0, 50)

    # 加图像标题
    # plt.title("MSN")

    # 加横纵坐标
    plt.xlabel("Error Threshold(m)")
    plt.ylabel("Location Accuarcy in the given Error Threshold(%)")

    # 图中加横纵线分割
    # plt.grid(True)

    # 颜色配色网站colordrop.io
    plt.plot(error_threshold, MP, color="#00a03e", marker='+', linewidth=1,  label='MP', markersize=8)
    plt.plot(error_threshold, HMMP, color="#f2317f", marker='*',  linewidth=1, label='HMMP', markersize=8)
    plt.plot(error_threshold, NNS_HMMP, color="#3399FF", marker='s', linewidth=1,  label='MSMN', markersize=8)
    # plt.plot(error_threshold, nns_N_12, color="#040000", marker='d',  linewidth=1, label='MSMN,N=12', markersize=8)
    # plt.plot(error_threshold, nns_N_16, color="#ea7070", marker='^',  linewidth=2, label='NNS-HMMP,N=16', markersize=8)

    # 加图注释
    # plt.legend(loc="best")
    plt.legend(loc='upper left', frameon=True)

    # Space plots a bit
    plt.subplots_adjust(hspace=0.25, wspace=0.40)

    plt.show()

def plot_sigma_change():
    # save error threshold and
    error_threshold = [0, 10, 20, 30, 40, 50]  # meter

    acc_sigma_05 = [0.562, 0.753, 0.863, 0.904, 0.927, 0.941]
    acc_sigma_1 = [0.721, 0.836, 0.927, 0.954, 0.968, 0.973]
    acc_sigma_2 = [0.785, 0.872, 0.927, 0.959, 0.968, 0.973]
    acc_sigma_5 = [0.845, 0.899, 0.936, 0.959, 0.968, 0.973]
    acc_sigma_10 = [0.868, 0.918, 0.941, 0.950, 0.963, 0.968]

    # 设置风格
    # plt.style.use('fivethirtyeight')
    # plt.style.use('ggplot')

    # 设置xtick坐标轴刻度线的方向，in,out,inout
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'

    fig, ax = plt.subplots()

    # this sets the figure on which we'll draw all the subplots
    # plt.figure(figsize=(10,8))

    # 设置横纵坐标轴范围
    ax.set_ylim(0.5, 1)
    ax.set_xlim(0, 50)

    # 加图像标题
    # plt.title("MSN")

    # 加横纵坐标
    plt.xlabel("Error Threshold(m)")
    plt.ylabel("Location Accuarcy in the given Error Threshold(%)")

    # 图中加横纵线分割
    # plt.grid(True)

    plt.plot(error_threshold, acc_sigma_05, color="#D74B4B", marker='+', linewidth=1, label='MSN,sigma=0.5', markersize=8)
    plt.plot(error_threshold, acc_sigma_1,  color="#DCDDD8",  marker='o', linewidth=1, label='MSN,sigma=1', markersize=12)
    plt.plot(error_threshold, acc_sigma_2,  color="#475F77", marker='v', linewidth=1, label='MSN,sigma=2', markersize=8)
    plt.plot(error_threshold, acc_sigma_5,  color="#354B5E", marker='1', linewidth=1, label='MSN,sigma=5', markersize=8)
    plt.plot(error_threshold, acc_sigma_10, color="#D0DFE6", marker='h', linewidth=1, label='MSN,sigma=10', markersize=8)

    # 加图注释
    # plt.legend(loc="best")
    # plt.legend(loc='upper left', frameon=True)
    plt.legend(loc='lower right', frameon=True)

    # Space plots a bit
    plt.subplots_adjust(hspace=0.25, wspace=0.40)

    plt.show()

def plot_NetVlad():
    error_threshold = [0, 10, 20, 30, 40, 50]  # meter

    # loc_threshold = [0.877, 0.950, 0.967, 0.983, 0.986, 0.989]
    net_location = [0.871, 0.890, 0.918, 0.955, 0.967, 0.967]
    nns_N_4 = [0.785, 0.872, 0.927, 0.959, 0.968, 0.973]
    without_nns_N_4 = [0.749, 0.831, 0.890, 0.909, 0.913, 0.922]

    # 设置风格
    # plt.style.use('fivethirtyeight')
    # plt.style.use('ggplot')
    # plt.style.use("presentation")
    # plt.style.use("presentation")
    # print(plt.style.available)

    # 设置xtick坐标轴刻度线的方向，in,out,inout
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'

    fig, ax = plt.subplots()

    # plt.figure(figsize=(20, 10))

    # this sets the figure on which we'll draw all the subplots
    # plt.figure(figsize=(10,8))

    # 设置横纵坐标轴范围
    ax.set_ylim(0.75, 1)
    ax.set_xlim(0, 50)

    # plt.upper_right_axis_ticks_on()

    # 加图像标题
    # plt.title("MSN")

    # 加横纵坐标
    plt.xlabel("Error Threshold(m)")
    plt.ylabel("Location Accuarcy in the given Error Threshold(%)")

    # 图中加横纵线分割
    plt.grid(True)

    plt.plot(error_threshold, nns_N_4, color="orange",  linewidth=2, label='NNS-HMMP', markersize=12)
    plt.plot(error_threshold, net_location, color="red", linewidth=2, label='NetVLAD', markersize=12)
    plt.plot(error_threshold, without_nns_N_4, color="puple", linewidth=2, label='HMMP', markersize=12)
    # plt.plot(error_threshold, acc_sigma_5, 'cD-', linewidth=2, label='NNS-HMMP,theta=5', markersize=12)
    # plt.plot(error_threshold, acc_sigma_10, 'y|-', linewidth=2, label='NNS-HMMP,theta=10', markersize=12)

    # 加图注释
    # plt.legend(loc="best")
    plt.legend(loc='upper left', frameon=True)

    # Space plots a bit
    plt.subplots_adjust(hspace=0.25, wspace=0.40)

    plt.show()

def plot_without_lstm():
    # save error threshold and
    error_threshold = [0, 10, 20, 30, 40, 50]  # meter

    nns_N_4 = [0.785, 0.872, 0.927, 0.959, 0.968, 0.973]
    nns_N_8 = [0.744, 0.836, 0.886, 0.941, 0.959, 0.963]
    ave_N_4 = [0.630, 0.780, 0.822, 0.854, 0.872, 0.890]
    ave_N_8 = [0.662, 0.808, 0.868, 0.913, 0.931, 0.945]

    # 设置风格
    # plt.style.use('fivethirtyeight')
    # plt.style.use('ggplot')

    # 设置xtick坐标轴刻度线的方向，in,out,inout
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'

    fig, ax = plt.subplots()

    # this sets the figure on which we'll draw all the subplots
    # plt.figure(figsize=(10,8))

    # 设置横纵坐标轴范围
    ax.set_ylim(0.6, 1)
    ax.set_xlim(0, 50)

    # 加图像标题
    # plt.title("MSN")

    # 加横纵坐标
    plt.xlabel("Error Threshold(m)")
    plt.ylabel("Location Accuarcy in the given Error Threshold(%)")

    # 图中加横纵线分割
    # plt.grid(True)

    plt.plot(error_threshold, nns_N_4, color="#ea7070", marker='+',linewidth=1,  label='MSMN,N=4', markersize=8)
    plt.plot(error_threshold, nns_N_8, color="#fdc4b6", marker='*', linewidth=1, label='MSMN,N=8', markersize=8)
    plt.plot(error_threshold, ave_N_4, color="#e59572", marker='|', linewidth=1,  label='MSMN w/o,N=4', markersize=8)
    plt.plot(error_threshold, ave_N_8, color="#2694ab", marker='^', linewidth=1, label='MSMN w/o,N=8', markersize=8)

    # 加图注释
    # plt.legend(loc="best")
    # plt.legend(loc='upper left', frameon=True)
    plt.legend(loc='lower right', frameon=True)

    # Space plots a bit
    plt.subplots_adjust(hspace=0.25, wspace=0.40)

    plt.show()

# def get_distance():
    GPS_Long_Lat_Compass = '/media/dyz/Data/Google_Street/Text/GPS_Long_Lat_Compass.mat'
    Cartesian_Location_Coordinates = '/media/dyz/Data/Google_Street/Text/Cartesian_Location_Coordinates.mat'
    gllc = scio.loadmat(GPS_Long_Lat_Compass)
    clc = scio.loadmat(Cartesian_Location_Coordinates)
    # print(gllc)
    print(len(clc["XYZ_Cartesian"]), clc["XYZ_Cartesian"])
    datas = clc["XYZ_Cartesian"][-100:]
    pre_data = datas[0]
    _x, _y, _z = [], [], []
    for data in datas[1:]:
        print(data)
        x = abs(pre_data[0] - data[0])
        y = abs(pre_data[1] - data[1])
        z = abs(pre_data[2] - data[2])
        print(x, y, z)
        _x.append(x)
        _y.append(y)
        _z.append(z)
        pre_data = data
    ave_x = sum(_x)/len(_x)
    ave_y = sum(_y)/len(_y)
    ave_z = sum(_z)/len(_z)
    print(sum(_x)/len(_x), sum(_y)/len(_y), sum(_z)/len(_z))
    # 1.63454975487 7.73565874614 8.25299155783
    print(np.sqrt(ave_x**2 + ave_y**2 + ave_z**2))

# get_distance()
# plot_acc_with_error_threshold()
# plot_N_change()
plot_sigma_change()
# plot_NetVlad()
# plot_without_lstm()
# plot_MP_HMMP()
