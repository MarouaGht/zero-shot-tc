import itertools
from sentence_transformers import InputExample, datasets,  models, LoggingHandler, SentenceTransformer, losses, evaluation
from PubmedEvaluator import PubmedTruePositiveEvaluator
from torch.utils.data import DataLoader
import pandas as pd
import os, math, sys, logging

def fit_models(
    num_folder, 
    nb_model,  
    input_csv_folder, 
    output_model_file, 
    batch, 
    s_bert_model, 
    validation_data
    ):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
    logging.info(f"input csv folder {input_csv_folder}")
    
    #Training data
    train_samples = []
    mesh_pos=[]
    mesh_neg = []
    abstracts=[]
    chunk = pd.read_csv(input_csv_folder, names = ['abstract','mesh_pos', 'mesh_neg'], chunksize=5000) 
    for chunk_data in chunk:
        mesh_pos.extend(chunk_data['mesh_pos'].values.tolist())
        mesh_neg.extend(chunk_data['mesh_neg'].values.tolist())
        abstracts.extend(chunk_data['abstract'].values.tolist())

    for i in range(len(abstracts)) :
                train_samples.append(InputExample(texts=[abstracts[i], mesh_pos[i],mesh_neg[i]]))
    loader = DataLoader(train_samples, shuffle=True, batch_size=batch)  


    validation=pd.read_csv(validation_data, names=['pmid','title','mesh_pos'])
    evaluator = PubmedTruePositiveEvaluator(validation, mesh_pos)

    epochs = 1
    warmup_steps = math.ceil(len(loader) * epochs / batch_size * 0.1) #10% of train data for warm-up

    if  num_folder == 0 :
        word_embedding_model = models.Transformer(s_bert_model, max_seq_length=350)
        pooler = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooler])
    else:
        model = SentenceTransformer(s_bert_model)

    loss = losses.TripletLoss(model,distance_metric=losses.TripletDistanceMetric.COSINE)
    logging.info(f"fit {nb_model} start")
    
    model.fit(
        train_objectives=  [(loader, loss)],
        evaluator=evaluator, #evalue chaque ev al step et à chaque epoch 
        epochs=epochs,
        #steps_per_epoch=2, #=nb_iteration if O, nb_iteration = nbligne / batch size = nb batch
        evaluation_steps=evaluation_steps, #evalue chaque 10 steps
        warmup_steps=warmup_steps,
        output_path= output_model_file + str(nb_model),
        show_progress_bar=True
    )  
    nb_model +=1

    logging.info("finish")

#bert = 'monologg/biobert_v1.1_pubmed'
output_model_file = './SSciFive/SSciFive_v'
evaluation_steps=450
#print((len(loader)/batch)/2)
batch_size = 64
nb_model = 0
training_data='all_triples.csv'
validation_data='evaluation.csv'
#bert = './SSciFive/1'
bert='razent/SciFive-base-Pubmed'
fit_models(0, 4, training_data, output_model_file, batch_size, bert, validation_data)
    