from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1


def get_resnet_model(device):
    resnet = InceptionResnetV1(
        classify=False,
        pretrained='vggface2'
    ).to(device)
    resnet.eval()
    return resnet


