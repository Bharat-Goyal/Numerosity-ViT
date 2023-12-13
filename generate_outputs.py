from utils import image_loading
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import ViTForImageClassification
import os 
from sklearn.metrics.pairwise import cosine_similarity
exp_directories = ['/home/hice1/bgoyal7/scratch/HML/experiment_data/exp1_equal_area_circles', 
    '/home/hice1/bgoyal7/scratch/HML/experiment_data/exp2_equal_circumference_circles',
    '/home/hice1/bgoyal7/scratch/HML/experiment_data/exp3_equal_area_diff_shapes',
    '/home/hice1/bgoyal7/scratch/HML/experiment_data/exp4_diff_area_diff_shapes',
    '/home/hice1/bgoyal7/scratch/HML/experiment_data/exp5_diff_area_diff_shapes_in_img', 
    '/home/hice1/bgoyal7/scratch/HML/experiment_data/exp6_google_images'
]

def get_cls_attention(outputs): 
    comp = torch.stack([torch.sum(i.cpu(), dim = 1) for i in outputs.attentions]) #Add attention across all heads 
    overall = comp.sum(dim=0) #Add attention across all layers
    imversion = np.kron(overall[:, 0, 1:].reshape((-1, 14, 14)), np.ones((1, 16, 16)))
    #OVERALL WILL HAVE THE ATTENTIONS FOR ALL TOKEN POSITIONS INCLUDING THE CLASSIFICATION TOKEN
    #OVERALL's SHAPE SHOULD BE : num of images x 197 x 197
    return imversion, overall

def save_outputs(input_dir, model):
    image_dict = image_loading.fetch_images(input_dir)
    def getNewPath(path):
        return '/'.join(input_dir.split('/')[:-2])  + f'/{path}/' + input_dir.split('/')[-1] + '/'
    att_map_path =getNewPath('ViT_maps')
    feat_path =getNewPath('ViT_embedding')
    att_path =getNewPath('ViT_attentions')
    # print(att_map_path, feat_path, att_path)
    #for embeddings we treat 1 experiment as one batch for ease of loading
    if not os.path.exists(att_path):
        os.makedirs(att_path)
    if not os.path.exists(feat_path):
        os.makedirs(feat_path)
    if not os.path.exists(att_map_path):
        os.makedirs(att_map_path)
    for (name, cur_experiment) in sorted(image_dict.items(), key=lambda x: x[0]):
        #WHEN LOADING THE HIDDEN_STATES/ATTENTION WEIGHTS, INDEXING WILL BE 0 BASED AND WILL CORRESPOND TO NUMEROSITY INDEX + 1
        with torch.no_grad():
            outputs = model(cur_experiment, output_attentions = True, output_hidden_states = True)
            imversion, overall_attention = get_cls_attention(outputs)
        cur_att_map_path = att_map_path + str(name) + '/'
        if not os.path.exists(cur_att_map_path):
            os.makedirs(cur_att_map_path)
        torch.save(outputs.hidden_states, feat_path + str(name) + '.pt')
        torch.save(overall_attention, att_path + str(name) + '.pt')
        for i in range(cur_experiment.shape[0]):
            # curimg = image_dict[subdir][i].cpu().numpy().transpose(1, 2, 0)
            curimg = cur_experiment[i].cpu().numpy().transpose(1, 2, 0)
            fig, ax = plt.subplots()
            ax.imshow(curimg)
            ax.imshow(imversion[i], alpha=0.5, cmap='gray')
            plt.savefig(cur_att_map_path + str(i+1) + '.png')
            plt.close()

def get_similarity(embedding_dir):
    def calculate_cosine_similarity(a_array, b_array):
        similarity_matrix = cosine_similarity(a_array, b_array)
        return np.sum(similarity_matrix)/a_array.shape[0] * b_array.shape[0]
    cosinedists  = {}
    for i in os.listdir(embedding_dir): 
        a = torch.load(os.path.join(embedding_dir, i), map_location=torch.device('cpu'))
        for j in os.listdir(embedding_dir): 
            b = torch.load(os.path.join(embedding_dir, j), map_location=torch.device('cpu'))
            cosinedists[(int(i.split('.')[0]), int(j.split('.')[0]))] = calculate_cosine_similarity(a[-1][:, 0, :], b[-1][:, 0, :])
    return cosinedists

def get_plots(embedding_dir, expname, save_dir): 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    def insert_into_dict(d, key, val):
        if key not in d: 
            d[key] = []
        d[key].append(val)
    def plot_for_dict(d, xlabel): 
        x, y = [], []
        for (i, val) in d.items():
            x.append(i)
            y.append(np.mean(val))
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_title(expname + ' ' + xlabel + ' effect')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cosine Similarity')
        plt.savefig(save_dir + '/' + expname + xlabel + '.png')
    
    bydist = {}
    byratio = {}
    bysize = {}
    cosinedists = get_similarity(embedding_dir)
    for (i, val) in cosinedists.items(): 
        insert_into_dict(byratio, max(i[0],i[1])/min(i[0],i[1]), val)
        insert_into_dict(bydist, abs(i[0] - i[1]), val)
        insert_into_dict(bysize, (i[0] + i[1])/2, val)
    plot_for_dict(bydist, 'Distance')
    plot_for_dict(byratio, 'Ratio')
    plot_for_dict(bysize, 'Size')

if __name__ == '__main__': 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    model.to(device)
    for i in exp_directories[-1:]:
        save_outputs(i, model)
