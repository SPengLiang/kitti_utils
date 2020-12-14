from sklearn.cluster import DBSCAN
import os
from tqdm import tqdm
import numpy as np
import cv2 as cv
# from . import calib_parse
# from . import project

import calib_parse
import project

def obtain_cluster_box(l_3d, l_2d, bbox2d, file_ind, P2):
    ind_first = l_3d[:, 2] > 0
    l_3d = l_3d[ind_first]
    l_2d = l_2d[ind_first]

    label = []
    for index, b in enumerate(bbox2d):
        print(b)
        if b[-1] < 0.7:
            continue
        bbox2d = cv.imread('/private/pengliang/KITTI3D/training/mask_rcnn/object_mask/{:0>6}_{}.png'.format(
            file_ind, index
        ), -1) / 256.

        bbox2d_2 = bbox2d.copy()
        bbox2d_2[bbox2d_2 < 0.7] = 0
        bbox2d_2[bbox2d_2 >= 0.7] = 1

        ind = bbox2d_2[l_2d[:, 1], l_2d[:, 0]].astype(np.bool)

        cam_points = l_3d[ind]

        if len(cam_points) < 10:
            continue

        cluster_index = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1).fit_predict(cam_points)

        cam_points = cam_points[cluster_index > -1]
        cluster_index = cluster_index[cluster_index > -1]

        if len(cam_points) < 10:
            continue

        cluster_set = set(cluster_index[cluster_index>-1])
        cluster_sum = np.array([len(cam_points[cluster_index == i]) for i in cluster_set])
        cam_points = cam_points[cluster_index == np.argmax(cluster_sum)]

        rect = cv.minAreaRect(np.array([(cam_points[:, [0, 2]]).astype(np.float32)]))
        (l_t_x, l_t_z), (w, l), rot = rect

        if w > l:
            w, l = l, w
            rot = 90 + rot

        if w > 1.5 and w < 2.1 and l > 3 and l < 4.5:
            rect = ((l_t_x, l_t_z), (w, l), rot)
            box = cv.boxPoints(rect)

            h = np.max(cam_points[:, 1]) - np.min(cam_points[:, 1])
            # y_center = (np.min(cam_points[:, 1]) + np.max(cam_points[:, 1])) / 2.
            y_center = np.mean(cam_points[:, 1])
            y = y_center + h / 2

            x, z = np.mean(box[:, 0]), np.mean(box[:, 1])
            Ry = -(np.pi / 2 - (-rot) / 180 * np.pi)

            c_3d = project.corner_3d([h, w, l, x, y, z, Ry])
            c_2d = project.convert_to_2d(c_3d, P2)
            bbox = [np.min(c_2d[:, 0]), np.min(c_2d[:, 1]),
                    np.max(c_2d[:, 0]), np.max(c_2d[:, 1])]

            slove = np.array([bbox[0], bbox[1], bbox[2], bbox[3],
                              h, w, l, np.mean(box[:, 0]), y, np.mean(box[:, 1]), Ry])
            slove = np.round(slove, 2)

            label.append(['Car', '-1', '-1', '0'] + list(slove))
    return np.array(label)



def obtain_cluster_box_for_raw_kitti(l_3d, l_2d, seg_bbox_path, seg_mask_path, file_ind, P2):
    ind_first = l_3d[:, 2] > 0
    l_3d = l_3d[ind_first]
    l_2d = l_2d[ind_first]
    bbox2d = np.loadtxt(seg_bbox_path).reshape(-1, 5)
    bbox_mask = (np.load(seg_mask_path))['masks']

    label = []
    for index, b in enumerate(bbox2d):
        if b[-1] < 0.7:
            continue

        bbox2d_2 = bbox_mask[index]
        bbox2d_2[bbox2d_2 < 0.7] = 0
        bbox2d_2[bbox2d_2 >= 0.7] = 1

        ind = bbox2d_2[l_2d[:, 1], l_2d[:, 0]].astype(np.bool)

        cam_points = l_3d[ind]

        if len(cam_points) < 10:
            continue

        cluster_index = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1).fit_predict(cam_points)

        cam_points = cam_points[cluster_index > -1]
        cluster_index = cluster_index[cluster_index > -1]

        if len(cam_points) < 10:
            continue

        cluster_set = set(cluster_index[cluster_index>-1])
        cluster_sum = np.array([len(cam_points[cluster_index == i]) for i in cluster_set])
        cam_points = cam_points[cluster_index == np.argmax(cluster_sum)]

        rect = cv.minAreaRect(np.array([(cam_points[:, [0, 2]]).astype(np.float32)]))
        (l_t_x, l_t_z), (w, l), rot = rect

        if w > l:
            w, l = l, w
            rot = 90 + rot

        if w > 1.5 and w < 2.1 and l > 3 and l < 4.5:
            rect = ((l_t_x, l_t_z), (w, l), rot)
            box = cv.boxPoints(rect)

            h = np.max(cam_points[:, 1]) - np.min(cam_points[:, 1])
            # y_center = (np.min(cam_points[:, 1]) + np.max(cam_points[:, 1])) / 2.
            y_center = np.mean(cam_points[:, 1])
            y = y_center + h / 2

            x, z = np.mean(box[:, 0]), np.mean(box[:, 1])
            Ry = -(np.pi / 2 - (-rot) / 180 * np.pi)

            c_3d = project.corner_3d([h, w, l, x, y, z, Ry])
            c_2d = project.convert_to_2d(c_3d, P2)
            bbox = [np.min(c_2d[:, 0]), np.min(c_2d[:, 1]),
                    np.max(c_2d[:, 0]), np.max(c_2d[:, 1])]

            slove = np.array([bbox[0], bbox[1], bbox[2], bbox[3],
                              h, w, l, np.mean(box[:, 0]), y, np.mean(box[:, 1]), Ry])
            slove = np.round(slove, 2)

            label.append(['Car', '-1', '-1', '0'] + list(slove))
    return np.array(label)

if __name__ == '__main__':

    # dst_dir = '/private/pengliang/M3D_RPN/data/kitti_split1/training/lidar_pred'
    # root_dir = '/private/pengliang/OCM3D/exp/lidar_pred'
    # train_file = np.loadtxt('/private/pengliang/Stereo_3D_Matching/train.txt', dtype=np.int32)
    # for i, v in tqdm(enumerate(train_file)):
    #     root_name = root_dir + '/{:0>6}.txt'.format(v)
    #     dst_name = dst_dir + '/{:0>6}.txt'.format(i)
    #     os.symlink(root_name, dst_name)
    # exit(0)

    # file_len = len(os.listdir('/private/pengliang/KITTI3D/training/label_2'))
    #
    # for i in tqdm(range(file_len)):
    #     lidar_depth = cv.imread('/private/pengliang/KITTI3D/training/lidar_depth/{:0>6}.png'.format(i), -1) / 256.
    #     calib_path = '/private/pengliang/KITTI3D/training/calib/{:0>6}.txt'.format(i)
    #     calib = calib_parse.parse_calib('3d', calib_path)
    #     l_3d, l_2d = project.convert_to_3d(lidar_depth, calib['P2'], 1, 0, 0)
    #
    #     if not os.path.exists(
    #             '/private/pengliang/KITTI3D/training/mask_rcnn/object_bbox/{:0>6}.txt'.format(i)):
    #         np.savetxt('/private/pengliang/OCM3D/exp/lidar_pred/{:0>6}.txt'.format(i), np.array([]), fmt='%s')
    #         continue
    #
    #     bbox2d = np.loadtxt('/private/pengliang/KITTI3D/training/mask_rcnn/object_bbox/{:0>6}.txt'.format(
    #         i
    #     )).reshape(-1, 5)
    #     if len(bbox2d) < 1:
    #         np.savetxt('/private/pengliang/OCM3D/exp/lidar_pred/{:0>6}.txt'.format(i), np.array([]), fmt='%s')
    #         continue
    #     # bbox2d = bbox2d[bbox2d[:, -1] > 0.7][:, :4]
    #     # if len(bbox2d) < 1:
    #     #     continue
    #
    #     label = obtain_cluster_box(l_3d, l_2d, bbox2d, i, calib['P2'])
    #     np.savetxt('/private/pengliang/OCM3D/exp/lidar_pred/{:0>6}.txt'.format(i), label, fmt='%s')


    eigen_file = np.loadtxt('/private/pengliang/bts/pytorch/bts_train_all.txt', dtype=str)
    l_list = eigen_file[:, 0]
    depth_list = eigen_file[:, 1]
    calib_list = np.array(['/private/pengliang/KITTI_raw/KITTI_raw/{}/calib_cam_to_cam.txt'.format(
        i.split('/')[5]) for i in l_list])

    for i, _ in enumerate(tqdm(l_list)):
        raw_lidar_path = depth_list[i].replace('groundtruth', 'velodyne').replace('png', 'npz')
        calib_path = depth_list[i].replace('groundtruth', 'velodyne').replace('png', 'npz')

        seg_mask_path = depth_list[i].replace('groundtruth', 'seg_mask').replace('png', 'npz')
        seg_bbox_path = depth_list[i].replace('groundtruth', 'seg_bbox').replace('png', 'txt')
        lidar_det_unsup_path = depth_list[i].replace('groundtruth', 'lidar_det_unsup').replace('png', 'txt')

        lidar_det_unsup_dir = os.path.dirname(lidar_det_unsup_path)
        if not os.path.exists(lidar_det_unsup_dir):
            os.makedirs(lidar_det_unsup_dir)

        lidar_depth = (np.load(raw_lidar_path))['velodyne_depth']
        calib = calib_parse.parse_calib('raw', [calib_list[i],
                                                calib_list[i].replace('calib_cam_to_cam',
                                                                      'calib_velo_to_cam')])
        l_3d, l_2d = project.convert_to_3d(lidar_depth, calib['P2'], 1, 0, 0)

        if not os.path.exists(seg_bbox_path):
            np.savetxt(lidar_det_unsup_path, np.array([]), fmt='%s')
            continue

        bbox2d = np.loadtxt(seg_bbox_path).reshape(-1, 5)
        if len(bbox2d) < 1:
            np.savetxt(lidar_det_unsup_path, np.array([]), fmt='%s')
            continue

        label = obtain_cluster_box_for_raw_kitti(l_3d, l_2d, seg_bbox_path,
                                                 seg_mask_path, i, calib['P2'])
        np.savetxt(lidar_det_unsup_path, label, fmt='%s')