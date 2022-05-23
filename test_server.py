from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import paired_cosine_distances
import json, pandas as pd
def nb_mesh_correct(mesh_pred, mesh_true):
    return str(len([mesh for mesh in mesh_pred if mesh in mesh_true]))+'/'+str(len(mesh_true))

def get_mesh_true(pmid, golden_data_path):
    with open (golden_data_path, 'r', encoding='UTF8') as f:
        pmid_mesh=json.load(f)
    
    return pmid_mesh[pmid]

def eval_results(results_file, golden_data_path):
    with open (results_file, 'r', encoding='UTF8') as f:
        results=json.load(f)
    pmid_metric={}
    for pmid in results.keys():
        mesh_true=get_mesh_true(pmid,golden_data_path)
        mesh_pred=results[pmid]
        pmid_metric[pmid]=nb_mesh_correct(mesh_pred,mesh_true)

    with open('nb_mesh corrects_v5_128_1109.json', mode='w', encoding='UTF8') as f:
        f.write(json.dumps(pmid_metric, indent=4))


def test(num_model):
    model = SentenceTransformer('SSciFive/SSciFive_v'+str(num_model))

    # Two lists of sentences
    pmid_mesh=pd.read_csv('../dataset_processed/pmid_article_unseen/1109_34727799_34757942.csv',header=None)
    pmid_mesh.columns=['pmid','article']
    articles = pmid_mesh['article'].tolist()
    '''
    with open('needed/mesh_seen.json',mode='r') as funseen:
                mesh_terms=json.load(funseen)

    with open('needed/mesh_unseen.json',mode='r') as funseen:
                #mesh_terms.append(json.load(funseen))
                #mesh_terms=json.load(funseen)

    '''
    with open('112_mesh.json',mode='r') as funseen:
                mesh_terms=json.load(funseen)

    #Compute embedding for both lists
    embeddings1 = model.encode(articles, show_progress_bar=True, convert_to_tensor=True)
    embeddings2 = model.encode(mesh_terms, show_progress_bar=True, convert_to_tensor=True)

    #Compute cosine-similarits
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    cos_sc={}
    #Output the pairs with their score
    for i in range(len(articles)):
        cos_sc[str(pmid_mesh['pmid'][i])]=[{str(mesh_terms[j]): cosine_scores[i][j].item()} for j in range(len(mesh_terms))]

    with open("../scores_finetuned/scores_finetuned_128_v"+str(num_model)+".json", 'a+',encoding='UTF8') as f:
        f.write(json.dumps(cos_sc, indent=4))

    with open("../scores_finetuned/scores_finetuned_128_v"+str(num_model)+".json", 'r',encoding='UTF8') as f:
        cos_sc=json.load(f)
    mesh_sim={}
    for pmid in cos_sc.keys():
        for mesh in cos_sc[pmid]:
            mesh_name=(list(mesh)[0])
            score=float(mesh[mesh_name])
            mesh_sim[mesh_name]=score
        
        cos_sc[pmid]={k: v for k, v in sorted(mesh_sim.items(), key=lambda item: item[1], reverse=True)[:20]}

    with open("scores_finetuned_sorted_128_v"+str(num_model)+".json", 'w',encoding='UTF8') as f:
        f.write(json.dumps(cos_sc, indent=4))
test(5)
golden_data= '../dataset_processed/pmid_mesh_unseen/1109_34727799_34757942.json'
results_file='scores_finetuned_sorted_128_v5.json'
eval_results(results_file,golden_data)
