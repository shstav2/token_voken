from src.img_utils.distance import get_distances, distance
from src.img_utils.resnet import get_interval_faces_embeddings, get_resnet_model


device = 'cuda:0'
data_dir = '/home/stav/Data/PATS_DATA/Videos/oliver/vU8dCYocuyI/216641_delete/FacesAll/FacesAll'
resnet_model = get_resnet_model(device)
frame_id_to_embeddings = get_interval_faces_embeddings(data_dir, device, resnet_model)
dist = get_distances(frame_id_to_embeddings)


# https://github.com/timesler/facenet-pytorch/blob/master/examples/lfw_evaluate.ipynb
from src.img_utils.reader import get_tensor_image, get_data_loader
# '/home/stav/Data/Sample/oliver/face.jpg'
from src.common.constants import OLIVER_FACE_PATH


# data = '/home/stav/Data/Sample'
# get_data_loader(data)

# np_oliver_face_image = get_tensor_image(OLIVER_FACE_PATH)

#
# oliver_face_embedding = resnet_model(np_oliver_face_image.unsqueeze(0).to(device))
# np_oliver_face_embedding = oliver_face_embedding.detach().to('cpu').numpy().reshape(-1)
#
# # /home/stav/Data/PATS_DATA/Videos/oliver/vU8dCYocuyI/216641_delete/FacesAll/FacesAll/00001/face_0.jpg
# e1 = frame_id_to_embeddings[1][0][1]
# # /home/stav/Data/PATS_DATA/Videos/oliver/vU8dCYocuyI/216641_delete/FacesAll/FacesAll/00001/face_1.jpg
# e2 = frame_id_to_embeddings[1][1][1]
# # /home/stav/Data/PATS_DATA/Videos/oliver/vU8dCYocuyI/216641_delete/FacesAll/FacesAll/00001/face_3.jpg
# e3 = frame_id_to_embeddings[1][2][1]
#
# e3_2 = frame_id_to_embeddings[100][2][1]
#
#
# distance(np_oliver_face_embedding, e1)