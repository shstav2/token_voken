import logging
import torch
from facenet_pytorch import MTCNN


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_mtcnn_model(device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(select_largest=False, thresholds=[0.9, 0.9, 0.9], device=device)
    logger.info(f'Init ({device}) MTCNN {mtcnn}..')
    return mtcnn
