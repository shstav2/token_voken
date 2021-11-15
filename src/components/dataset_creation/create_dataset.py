
# CHECK VOKENS ARE THE SAME IN ALL FILES:
# ========================================
"""
import numpy as np
import pandas as pd
from src.common.path_resolvers import resolve_interval_facial_embedding_path
# ---- np array
vokens_path = '/home/stav/Data/Vokenization/Datasets/Oliver_V1/vokens.npy'
vokens = np.load(vokens_path)
# ---- dataframe
token_voken_path = '/home/stav/Data/Vokenization/Datasets/Oliver_V1/df_token_voken.csv'
df_token_voken = pd.read_csv(token_voken_path)
sample = df_token_voken.sample().iloc[0]
voken_str = sample[COL_VOKEN]
voken_str = voken_str.replace('\n', '').replace('[', '').replace(']', '')
voken_df = np.fromstring(voken_str, sep=' ')
# ---- FECNet - '/home/stav/Data/PATS_DATA/Videos/oliver/iAgKHSNqxa8/214675/FECNet/00111.npy'
fecnet_path = resolve_interval_facial_embedding_path(str(sample.interval_id), sample.selected_frame)
voken_fecnet = np.load(fecnet_path)
np.isclose(voken_fecnet, voken_df)
np.isclose(voken_fecnet, vokens[sample.voken_id - 1])
"""

# CHECK VOKENS ARE THE SAME IN ALL FILES:
# ========================================
"""
vokens_v1_path = '/home/stav/Data/Vokenization/Datasets/Oliver_V1/vokens.npy'
vokens_v1 = np.load(vokens_v1_path)
vokens_v2_path = '/home/stav/Data/Vokenization/Datasets/Oliver_V2/vokens.npy'
vokens_v2 = np.load(vokens_v2_path)
>>> np.array_equal(vokens_v1, vokens_v2)
True
>>> (vokens_v1 == vokens_v2).all(axis=1).sum()
76873
>>> vokens_v1.shape[0] - _
4839 # 6%
"""

"""
df_token_voken[df_token_voken['interval_id'] == int(interval_id)]
vokens[78460]
np.load(resolve_interval_facial_embedding_path(interval_id, 38))
"""


"""
import numpy as np
import pandas as pd
path = '/home/stav/Data/Vokenization/Datasets/Noah_V1/df_token_voken_pkl.csv'
df_token_voken = pd.read_pickle(path) 
vokens = df_token_voken['voken'].tolist()
np_vokens = np.stack(vokens)
"""