import gzip, json, os, csv

def firstBefore_Chars(x):
    return(int(x.split("_")[0]))

def start_index(x):
    return(int(x.split("_")[1]))

def end_index(x):
    return(int(x.split("_")[2].split(".")[0]))

def parsed_pubmed_dataset(file_path):
    dataset = []
    with gzip.open(file_path, 'rt', encoding='utf8') as fd:
        for line in fd:
            try:
                splits = line.strip().split("\t")
                pmid = int(splits[0])
                text = splits[1]
                mesh_terms = splits[2:]
                dataset.append((pmid, text, mesh_terms))
            except:
                pass
    return dataset

def generate_new_triples_from_pairs(dataset, dataset_path):
    pmid_mesh_all_files = sorted(os.listdir(dataset_path+'new_pos_neg'), key = firstBefore_Chars) 
    f_evaluation_data = open(dataset_path + 'training_data1.csv' ,'a')
    writer = csv.writer(f_evaluation_data)
    for data_line in range(3000000, len(dataset)):
        pmid = dataset[data_line][0]
        for i in range(len(pmid_mesh_all_files)):     
            if  start_index(pmid_mesh_all_files[i] ) <= pmid <= end_index(pmid_mesh_all_files[i] ) :
                print(pmid)
                pmid_mesh_all=dataset_path+'new_pos_neg/'+pmid_mesh_all_files[i] 
                with open(pmid_mesh_all, mode= 'r', encoding="UTF8") as f_pmid_mesh_all:
                    pmid_mesh = json.load(f_pmid_mesh_all)
                mesh_neg = pmid_mesh[str(pmid)]['mesh_neg']
                mesh_pos =  dataset[data_line][2]
                cpt_mesh_pos =0
                for mesh in mesh_pos :
                    if cpt_mesh_pos == 0:
                        mesh_pos_list =  mesh
                    else :
                        mesh_pos_list = mesh_pos_list + ';' + mesh
                    cpt_mesh_pos +=1
                cpt_mesh_neg =0       
                for mesh in mesh_neg :
                    if cpt_mesh_neg == 0:
                        mesh_neg_list =  mesh
                    else :
                        mesh_neg_list = mesh_neg_list + ';' + mesh
                    cpt_mesh_neg +=1
                    if cpt_mesh_neg == cpt_mesh_pos : break
                writer.writerow([pmid,dataset[data_line][1], mesh_pos_list, mesh_neg_list ])
                continue

def main():
    file_path = "./said_data/data1.zip"
    dataset = parsed_pubmed_dataset(file_path)
    print(len(dataset))
    dataset_path="./said_data/"
    generate_new_triples_from_pairs(dataset, dataset_path)

if __name__ == "__main__":
        main()