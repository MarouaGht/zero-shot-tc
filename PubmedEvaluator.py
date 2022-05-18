import csv, time,sys, os
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers import util
import logging
logger = logging.getLogger(__name__)

class PubmedTruePositiveEvaluator(SentenceEvaluator):
    def __init__(
        self, 
        validation_data, 
        mesh_terms, 
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = True,
        write_csv: bool = True,
        top_ks=[1, 2, 5, 10, 20, 50, 100]
    ):
        self.pmids=validation_data['pmid'].values.tolist()
        self.titles=validation_data['title'].values.tolist()
        self.mesh_terms=mesh_terms
        self.mesh_pos=self.get_mesh_pos(validation_data)
        
        self.mesh_terms.extend(self.mesh_pos)
        self.mesh_terms=list(set(self.mesh_terms))
        self.pmid_mesh=self.get_pmid_mesh(validation_data)
        self.name=name
        self.batch_size=batch_size
        self.top_ks=top_ks
        self.start = 0
        self.true_positives={k:0 for k in self.top_ks} #map where keys = top_k and values = number of true positives

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.stdout = csv.writer(sys.stdout)
        
        self.csv_headers = ['epoch', 'step'] + [str(top_k) for top_k in top_ks]
        self.csv_file: str = "pubmed_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.write_csv = write_csv


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("PubmedTruePositiveEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        if self.start != 0:
            self.end = time.time()
            print(f'{(self.end - self.start):.1f} seconds for training steps')
        start = time.time()

        title_embeddings=model.encode(self.titles,self.batch_size, show_progress_bar=True, convert_to_tensor=True)
        mesh_embeddings=model.encode(self.mesh_terms,self.batch_size, show_progress_bar=True, convert_to_tensor=True)

        cosine_scores = util.cos_sim(title_embeddings, mesh_embeddings)

        for top_k in self.top_ks:
            pred_positives=self.sort_cosine_scores(cosine_scores,top_k)     
            self.compute_true_positives(pred_positives,top_k)
            logger.info("Number of true positives :   \t{:.2f}/".format(self.true_positives[top_k]))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode='a' if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)
                writer.writerow([epoch, steps] + [self.true_positives[top_k] for top_k in self.top_ks])

        end = time.time()
        print(f'{(end - start):.1f} seconds for evaluation')
        self.start = time.time()
        logger.info("Number of mesh_terms :   \t{:.2f}/".format(len(self.mesh_terms)))
        return self.true_positives[self.top_ks[-1]]/len(self.mesh_terms)
    

    def true_positives_per_article(self,mesh_pred, mesh_true):
        return len([mesh for mesh in mesh_pred if mesh in mesh_true])

    def compute_true_positives(self,results, top_k):
        pmid_metric={}
        for pmid in results.keys():
            mesh_true=self.pmid_mesh[pmid]
            mesh_pred=results[pmid]
            pmid_metric[pmid]=self.true_positives_per_article(mesh_pred,mesh_true)
            self.true_positives[top_k]+=pmid_metric[pmid]


    def sort_cosine_scores(self,cos_sc,top_k):
        cosine_scores={}
        for i in range(len(self.titles)):
            pmid=str(self.pmids[i])
            cosine_scores[pmid]=[
                {str(self.mesh_terms[j]): cos_sc[i][j].item()} 
                for j in range(len(self.mesh_terms))]
        mesh_sim={}
        for pmid in cosine_scores.keys():
            for mesh in cosine_scores[pmid]:
                mesh_name=(list(mesh)[0])
                score=float(mesh[mesh_name])
                if score>=0.5:
                    mesh_sim[mesh_name]=score
            
            cosine_scores[pmid]={k: v for k, v in sorted(mesh_sim.items(), key=lambda item: item[1], reverse=True)[:top_k]}
        return cosine_scores

    def get_pmid_mesh(self,validation_data):
        pmid_mesh={}
        for i in range(len(validation_data.index)):
            pmid=str(validation_data['pmid'][i])
            mesh_pos=validation_data['mesh_pos'][i].split(';')
            pmid_mesh[pmid]=mesh_pos
        return pmid_mesh

    def get_mesh_pos(self,validation_data):
        list_mesh=validation_data['mesh_pos'].values.tolist()
        mesh_pos= []
        for line in list_mesh:
            print(line)
            mesh_pos.extend(line.split(';'))
        return mesh_pos
