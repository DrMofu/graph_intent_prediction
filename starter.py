from models import *
from setting import *
from handle_dataset import *

# get dataset
# handle differently for different dataset. (You need to specify handle_dataset.py for different format of dataset)
[train_data,dev_data,test_data],citation,e2id,r2id = get_dataset()


# build graph
src, rel, dst = [], [], []
info_num, train_num, test_num = 0, 0, 0
for pair in tqdm(citation, total=len(citation)):
    info_num += 1
    src.append(e2id[pair[0]])
    dst.append(e2id[pair[1]])
    rel.append(r2id[pair[2]])

for pair in tqdm(train_data, total=len(train_data)):
    train_num += 1
    src.append(e2id[pair[0]])
    dst.append(e2id[pair[1]])
    rel.append(r2id[pair[2]])
    
for pair in tqdm(test_data, total=len(test_data)):
    test_num += 1
    src.append(e2id[pair[0]])
    dst.append(e2id[pair[1]])    
    rel.append(r2id[pair[2]])

g = dgl.graph((src, dst), num_nodes=len(e2id), idtype=th.int64)
g.edata['rel_type'] = th.tensor(rel)

# build sampler
train_sampler = dgl.dataloading.NeighborSampler(FANOUTS)
train_sampler = dgl.dataloading.as_edge_prediction_sampler(
        train_sampler,
        exclude='self',
        negative_sampler=None)
train_dataloader = dgl.dataloading.DataLoader(
    g,
    np.arange(info_num,info_num+train_num),
    train_sampler,
    device=device,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    num_workers=16)

test_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)  # dgl.dataloading.NeighborSampler(FANOUTS)
test_sampler = dgl.dataloading.as_edge_prediction_sampler(
    test_sampler,
    exclude='self',
    negative_sampler=None)
test_dataloader = dgl.dataloading.DataLoader(
    g,
    np.arange(info_num + train_num, info_num + train_num + test_num),
    test_sampler,
    device=device,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=16)


# init model
model = LinkPredictionModel(len(e2id), IN_DIM, OUT_DIM,score_func="complex").to(device)
opt = th.optim.Adam(model.parameters(), lr=LR)
scheduler = th.optim.lr_scheduler.MultiStepLR(opt, milestones=[1, 3, 5], gamma=0.1)

# train & test
for i in range(EPOCHS):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_dataloader)
    for j, (_, positive_graph, blocks) in enumerate(pbar, 1):
#         blocks = [b.to(device) for b in blocks]
#         positive_graph = positive_graph.to(device)
        
        positive_graph.edata["rel_type"] = positive_graph.edata["rel_type"]
        pos_score, neg_score = model(positive_graph, blocks)
        loss = ce_loss(pos_score, neg_score)
        total_loss += loss.detach().cpu()
        pbar.set_postfix({'total_loss': total_loss / j})

        opt.zero_grad()
        loss.backward()
        opt.step()

    if (i + 1) % EVAL_EPOCHS == 0:
        with th.no_grad():
            model.eval()
            predict_y = []
            true_y = []
            pbar = tqdm(test_dataloader, total=len(
                test_data) // EVAL_BATCH_SIZE)
            for _, positive_graph, blocks in pbar:
                positive_graph.edata["rel_type"] = positive_graph.edata["rel_type"]
                predict_y.extend(model.refer(positive_graph, blocks).tolist())
                true_y.extend((positive_graph.edata["rel_type"]).tolist())
            print("ACC:[{:.2f}] | P:[{:.2f}] | R:[{:.2f}] | F1:[{:.2f}]".format(100*accuracy_score(true_y,predict_y),100*precision_score(true_y,predict_y,average="macro"),
            100*recall_score(true_y,predict_y,average="macro"),100*f1_score(true_y,predict_y,average="macro")))

    scheduler.step()