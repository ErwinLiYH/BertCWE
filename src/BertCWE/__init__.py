import transformers
import torch
from tqdm import tqdm
import os
from Kkit import prj_control
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# cache_path
# ---vecs_file1
# ---vecs_file2
# ---......

class BertModel:
    def __init__(self, MODEL = "bert-base-uncased", TOKENIZER = "bert-base-uncased"):
        self.model = transformers.BertForMaskedLM.from_pretrained(MODEL)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(TOKENIZER)
    
    def create_bert_vectors(self, target_voca, comments, cache_path, layer=0):
        for comment in tqdm(comments):
            self.__bert_parse_comment(self, target_voca=target_voca, comment=comment, cache_path=cache_path, layer=layer)
        result_dic = overview_vecs(cache_path, False)
        for i in [j for j in target_voca if i not in list(result_dic.keys())]:
            print("%s not in comments, transfer it by itself directly"%i)
            self.__bert_parse_comment(self, target_voca=target_voca, comment=i, cache_path=cache_path, layer=layer)

    def __bert_parse_comment(self, target_voca, comment, cache_path, layer):
        encoded = self.tokenizer.encode_plus(comment, return_tensors="pt", return_token_type_ids = False, return_attention_mask = False)
        input_ids = encoded.input_ids.squeeze()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        loca_dict = get_loca_from_ids(tokens, target_voca)

        raw_vectors = self.model(input_ids.unsqueeze(dim=0), output_hidden_states = True).hidden_states[layer]
        vec_dic = {}
        for k,v in loca_dict.items():
            vec = raw_vectors[0][v[0]:v[-1]+1]
            if vec.shape[0] != 1:
                vec = torch.mean(vec, dim=0)
            else:
                vec = vec.squeeze()
            vec = vec.detach().numpy()
            vec_dic[k] = vec
        store_vec(vec_dic, cache_path)

def store_vec(vec_dic, cache_path):
    cache_list = []
    try:
        cache_list = os.listdir(cache_path)
    except:
        os.mkdir(cache_path)
    for k,v in vec_dic.items():
        if k in cache_list:
            l = prj_control.load_result(os.path.join(cache_path, k))
            l.append(vec_dic[k])
            prj_control.store_result(os.path.join(cache_path, k), l)
        else:
            prj_control.store_result(os.path.join(cache_path, k), [vec_dic[k]])          

def get_loca_from_ids(input_ids, tokens):
    locations = []
    voca = []
    len = 0
    for index, item in enumerate(input_ids[1:-1]):
        index += 1
        if item.startswith("##") == False:
            locations.append([index])
            voca.append(item)
            len = 0
        else:
            len+=1
            i = locations.index(list(range(index-len,index)))
            locations[i].append(index)
            voca[i] = voca[i]+item.lstrip("##")
    result = {}
    for i in tokens:
        try:
            result[i] = locations[voca.index(i)]
        except:
            pass
    return result

def overview_vecs(cache_path, verbose=True):
    cache_list = os.listdir(cache_path)
    num_of_vecs = []
    shape_list = []
    result_dic = {}
    for i in tqdm(cache_list):
        l = prj_control.load_result(os.path.join(cache_path, i))
        for j in l:
            if j.shape not in shape_list:
                shape_list.append(j.shape)
        num_of_vecs.append(len(l))
    info_str = "total vocabulary: %d\ntotal vectors: %d\n----------------\n"%(len(cache_list), sum(num_of_vecs))
    for i in range(len(cache_list)-1):
        info_str+="%s: %d\n"%(cache_list[i], num_of_vecs[i])
        result_dic[cache_list[i]] = num_of_vecs[i]
    info_str+="%s: %d\n"%(cache_list[-1], num_of_vecs[-1])
    result_dic[cache_list[-1]] = num_of_vecs[-1]
    if verbose:
        print(info_str)
        print("shepe of vectors:")
        print(shape_list)
    return result_dic

def visualize_vectors(vecs_file_path, save_path=None):
    vecs = prj_control.load_result(vecs_file_path)
    vecs = np.stack(vecs)
    pca = PCA(n_components=2,svd_solver='full')
    PCA_res = pca.fit_transform(vecs)
    XY_range = (np.amin(PCA_res), np.amax(PCA_res))
    plt.figure(figsize=(10, 10))
    plt.scatter(PCA_res[:,0], PCA_res[:,1])
    plt.xlim = XY_range
    plt.ylim = XY_range
    plt.show()
    if save_path != None:
        plt.savefig(save_path)

def collect_data(cache_path, store_path="./results"):
    labels = []
    final_vecs = []
    cache_list = os.listdir(cache_path)
    for i in tqdm(cache_list):
        l = prj_control.load_result(os.path.join(cache_path, i))
        for j in l:
            final_vecs.append(j)
            labels.append(i)
    final_vecs = np.stack(final_vecs)
    prj_control.store_result(os.path.join(store_path, "labels_bert_all"), labels)
    print("save all vectors to %s"%os.path.join(store_path, "labels__bert_all"))
    prj_control.store_result(os.path.join(store_path, "final_vecs_bert_all"), final_vecs)
    print("save labels to %s"%os.path.join(store_path, "final_vecs_bert_all"))


def compress_vectors(cache_path, store_path="./results", n = 10):
    cache_list = []
    try:
        cache_list = os.listdir(cache_path)
    except:
        os.mkdir(cache_path)
    labels = []
    final_vecs = []
    if n == 1:
        for i in tqdm(cache_list):
            vecs = prj_control.load_result(os.path.join(cache_path, i))
            vecs = np.stack(vecs)
            vec = np.mean(vecs, axis=0)
            final_vecs.append(vec)
            labels.append(i)
    else:
        cluster = AgglomerativeClustering(n_clusters = n)
        for i in tqdm(cache_list):
            vecs = prj_control.load_result(os.path.join(cache_path, i))
            vecs = np.stack(vecs)
            cluster_label = cluster.fit_predict(vecs)
            for j in range(n):
                temp_vec = vecs[np.where(cluster_label == j)]
                temp_vec = np.mean(temp_vec, axis=0)
                final_vecs.append(temp_vec)
                labels.append(i)

    final_vecs = np.stack(final_vecs)
    prj_control.store_result(os.path.join(store_path, "labels_bert_%d"%n), labels)
    print("save all vectors to %s"%os.path.join(store_path, "labels_bert_%d"%n))
    prj_control.store_result(os.path.join(store_path, "vecs_bert_%d"%n), final_vecs)
    print("save labels to %s"%os.path.join(store_path, "vecs_bert_%d"%n))

def PC_vectors(cache_path, store_path="./results", n = 1):
    cache_list = []
    try:
        cache_list = os.listdir(cache_path)
    except:
        os.mkdir(cache_path)
    labels = []
    final_vecs = []

    for i in tqdm(cache_list):
        vecs = prj_control.load_result(os.path.join(cache_path, i))
        vecs = np.stack(vecs)

        pca = PCA(n_components=n)
        vec = pca.fit_transform(vecs.T).T
        vec = np.squeeze(vec)

        final_vecs.append(vec)
        labels.append(i)

    final_vecs = np.stack(final_vecs)
    prj_control.store_result(os.path.join(store_path, "labels_bert_PC%d"%n), labels)
    print("save all vectors to %s"%os.path.join(store_path, "labels_bert_PC%d"%n))
    prj_control.store_result(os.path.join(store_path, "vecs_bert_PC%d"%n), final_vecs)
    print("save labels to %s"%os.path.join(store_path, "vecs_bert_PC%d"%n))

# get_loca_from_ids(['[CLS]', 'i', 'love', '##ddd', '##asd', 'tf', 'you', '##are', '[SEP]'], ['i', 'love', 'lovedddasd', 'tf', 'youare'])