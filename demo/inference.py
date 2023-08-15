import argparse
import glob
import multiprocessing as mp
import os
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import time
import pandas as pd
import cv2
import torch
import tqdm
#from symmetry import *
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from extract import *
from symmetry import *
from patches import  *
from metrics import *
from testing import *
from predictor import VisualizationDemo
from adet.config import get_cfg

# constants
WINDOW_NAME = "COCO detections"
SYM_MODEL_PATH="/content/drive/MyDrive/ReID/siam-tr-state-dict.pt"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    # cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    # cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--masks-path", nargs="+",help="path to the mask folder.")
    parser.add_argument("--images-path",nargs="+", help="path to the image folder.")
    parser.add_argument("--input", nargs="+", help="using pandas read the csv file ")

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    #model=torch.load('/content/drive/MyDrive/ReID/siam-tr-model.pt').cuda()
    if args.input:
        df=pd.read_csv(args.input[0])
        img_path="/content/drive/MyDrive/ReID/Samples/UFPR"
        msk_path="/content/drive/MyDrive/ReID/Samples/UFPR"
        model=fetch_symmetry_model(SYM_MODEL_PATH)
        acc_score,char_error=evaluate_without_siamese(df,img_path,msk_path,demo)
        print("accuracy and cer without symmetry",acc_score,char_error)
        print("accuracy and cer with symmetry",evaluate_with_siames(df,img_path,msk_path,demo,model))


        