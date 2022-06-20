from yacs.config import CfgNode as CN 

_C = CN()

_C.CAMERA_NAME_PREFIX = 'cam_'

_C.SERIAL_NUMBERS = [
    '023322060722', # right corner
    '045422061046', # front (right)
    '810512062562', # left corner
    '950122061188', # front (left - 415)
]

_C.RGB_LCM_TOPIC_NAME_SUFFIX = 'rgb_image'
_C.DEPTH_LCM_TOPIC_NAME_SUFFIX = 'depth_image'
_C.INFO_LCM_TOPIC_NAME_SUFFIX = 'info'
_C.POSE_LCM_TOPIC_NAME_SUFFIX = 'pose'

_C.WIDTH = 640
_C.HEIGHT = 480
_C.FRAME_RATE = 30

def get_default_multi_realsense_cfg():
    return _C.clone()
