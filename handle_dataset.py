from setting import *

def get_dataset():
    def clean_dataset(input_df):
        scicite_citation_pair = collections.defaultdict(set)
        for item in zip(input_df["citingPaperId"],input_df["citedPaperId"],input_df["label"]):
            if item[0]!="None" and item[1]!="None":
                str_hash = (item[0],item[1])
                scicite_citation_pair[str_hash].add(item[2])
        return_ans = []
        for key in scicite_citation_pair:
            if key[0] not in e2id or key[1] not in e2id: # ignore this citation pair, because you can't find any record for those papers.
                continue
            if len(scicite_citation_pair[key])>0:
                for intent in list(scicite_citation_pair[key]):
                    return_ans.append([key[0],key[1],intent])
        return return_ans
    # read data
    with open("/nas/ckgfs/pubgraphs/xinweidu/intent/crawling/api_data/backup/other_cite_this_info.pkl","rb") as f:
        other_cite_this_info = pickle.load(f)
        
    with open("/nas/ckgfs/pubgraphs/xinweidu/intent/crawling/api_data/backup/this_cite_other_info.pkl","rb") as f:
        this_cite_other_info = pickle.load(f)

    citation = []
    for paper in this_cite_other_info:
        for infos in this_cite_other_info[paper]:
            for intent in infos[1]:
                citation.append([paper,infos[0],intent])
                
    for paper in other_cite_this_info:
        for infos in other_cite_this_info[paper]:
            for intent in infos[1]:
                citation.append([infos[0],paper,intent])

    paper_sci_id = set()
    for cite in citation:
        paper_sci_id.add(cite[0])
        paper_sci_id.add(cite[1])
    paper_sci_id = list(paper_sci_id)

    e2id = {}
    r2id = {}

    for i, entity in enumerate(paper_sci_id):
        e2id[entity] = i
        
    r2id = {'background': 0, 'methodology': 1, 'result': 2, 'method':1}

    sci_train_df = pd.read_csv("/nas/ckgfs/pubgraphs/xinweidu/intent/sci_dataset/scicite/intermediate/sci_train.csv")
    sci_test_df = pd.read_csv("/nas/ckgfs/pubgraphs/xinweidu/intent/sci_dataset/scicite/intermediate/sci_test.csv")
    sci_dev_df = pd.read_csv("/nas/ckgfs/pubgraphs/xinweidu/intent/sci_dataset/scicite/intermediate/sci_dev.csv")

    train_data = clean_dataset(sci_train_df)
    dev_data = clean_dataset(sci_dev_df)
    test_data = clean_dataset(sci_test_df)

    return [train_data,dev_data,test_data],citation,e2id,r2id



    
