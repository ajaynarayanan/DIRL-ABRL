import os, sys, os.path as osp
import gen_utils
import numpy as np
from re_ranking_feature import re_ranking

def evaluate(feature_dir, prefix, result_dir, settings):
    """to evaluate and rank results for SYSU_MM01 dataset
    
    Arguments:
        feature_dir {str} -- a dir where features are saved
        prefix {str} -- prefix of file names
        result_dir {str} -- a directory to save results
        settings {dict} -- a dictionary specifying test mode and other test settings 
    """
    gallery_cams, probe_cams = gen_utils.get_cam_settings(settings)
    all_cams = list(set(gallery_cams + probe_cams))  # get unique cams
    features = {}

    # get permutation indices
    cam_permutations = gen_utils.get_cam_permutation_indices(all_cams)
  
    # get test ids
    test_ids = gen_utils.get_test_ids(settings["test_mat_name"])
    for cam_index in all_cams:
        # read features
        cam_feature_file = osp.join(feature_dir, (prefix + "_cam{}").format(cam_index))
        features["cam" + str(cam_index)] = gen_utils.load_feature_file(cam_feature_file)

    # perform testing
    print(list(features.keys()))
    cam_id_locations = [1, 2, 2, 4, 5, 6]
    # camera 2 and 3 are in the same location
    mAPs = []
    cmcs = []

    for run_index in range(10):
        print("trial #{}".format(run_index))
        X_gallery, Y_gallery, cam_gallery, X_probe, Y_probe, cam_probe = gen_utils.get_testing_set(
            features,
            cam_permutations,
            test_ids,
            run_index,
            cam_id_locations,
            gallery_cams,
            probe_cams,
            settings,
        )

        # print(X_gallery.shape, Y_gallery.shape, cam_gallery.shape, X_probe.shape, Y_probe.shape, cam_probe.shape)
        if settings["rerank"]:
            dist = re_ranking(X_probe, X_gallery, settings["k1"], settings["k2"], settings["lambda_value"])
        else:
            dist = gen_utils.euclidean_dist(X_probe, X_gallery)
        cmc = gen_utils.get_cmc_multi_cam(
            Y_gallery, cam_gallery, Y_probe, cam_probe, dist
        )
        mAP = gen_utils.get_mAP_multi_cam(
            Y_gallery, cam_gallery, Y_probe, cam_probe, dist
        )

        print("rank 1 5 10 20", cmc[[0, 4, 9, 19]])
        print("mAP", mAP)
        cmcs.append(cmc)
        mAPs.append(mAP)

    # find mean mAP and cmc
    cmcs = np.array(cmcs)  # 10 x #gallery
    mAPs = np.array(mAPs)  # 10
    mean_cmc = np.mean(cmcs, axis=0)
    mean_mAP = np.mean(mAPs)
    print("mean rank 1 5 10 20", mean_cmc[[0, 4, 9, 19]])
    print("mean mAP", mean_mAP)

def evaluate_results(feature_dir, prefix, mode, number_shot, total_runs=10, result_dir="result", type_= "overall", rerank = False, k1 = 20, k2= 4, lambda_value = 0.3, test_mat_name ="test"):

    # evaluation settings
    print(
        "running evaluation for features from {} with prefix {} with mode {} and number shot {}".format(
            feature_dir, prefix, mode, number_shot
        )
    )
    print('total test runs:', total_runs)
    settings = {}
    settings["mode"] = mode  # indoor | all
    settings["number_shot"] = number_shot  # 1 = single-shot | 10 = multi-shot
    settings["rerank"] = rerank
    settings["k1"] = k1
    settings["k2"] = k2
    settings["lambda_value"] = lambda_value
    settings["test_mat_name"] = test_mat_name 
    evaluate(feature_dir, prefix, result_dir, settings)    



if __name__ == "__main__":
    feature_dir = "./epoch_50"
    prefix = "epoch_50"
    result_dir = "./result"

    # evaluation settings
    settings = {}
    settings["mode"] = "indoor"         # indoor | all
    settings["number_shot"] = 1         # 1 = single-shot | 10 = multi-shot
    settings["rerank"] = False          # Reranking params
    settings["k1"] = 20
    settings["k2"] = 4
    settings["lambda_value"] = 0.3
    settings["test_mat_name"] = "test"  # val = testing on validation dataset
    evaluate(feature_dir, prefix, result_dir, settings)
