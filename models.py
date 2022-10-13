from setting import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(max_len, d_model)
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: th.Tensor) -> th.Tensor:
        '''
        Args:
            x: Tensor, shape [seq_len, ]
        '''
        return th.index_select(self.pe, 0, x)

class TGATEncoding(nn.Module):
    def __init__(self, d_model: int):
        super(TGATEncoding, self).__init__()

        self.freq = nn.Parameter((th.from_numpy(1 / 10 ** np.linspace(0, 9, d_model))).float())
        self.phase = nn.Parameter(th.zeros(d_model).float())

    def forward(self, x):
        harmonic = th.cos(x.view(-1, 1) * self.freq.view(1, -1) + self.phase.view(1, -1))
        return harmonic

    
def u_add_e(src_u, src_e, dst_e):
    def fn(edges):
        return {dst_e: edges.src[src_u] + edges.data[src_e]}
    return fn


def u_cat_e(src_u, src_e, dst_e):
    def fn(edges):
        return {dst_e: th.cat([edges.src[src_u], edges.data[src_e]], dim=-1)}
    return fn


def u_add_mul_e(src_u, src_e, w_e, dst_e):
    def fn(edges):
        return {dst_e: (edges.src[src_u] + edges.data[src_e]) * edges.data[w_e]}
    return fn


def u_cat_mul_e(src_u, src_e, w_e, dst_e):
    def fn(edges):
        return {dst_e: th.cat([edges.src[src_u], edges.data[src_e]], dim=-1) * edges.data[w_e]}
    return fn


class CustomSAGEConv(dgl.nn.SAGEConv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(CustomSAGEConv, self).__init__(in_feats, out_feats,
                                             aggregator_type, feat_drop, bias, norm, activation)

        if 'att' in TEMPORAL_VARIATION:
            self._lambda = nn.Parameter(th.Tensor(in_feats if T_DIM else 1))
            self._beta = nn.Parameter(th.Tensor(in_feats if T_DIM else 1))

            nn.init.ones_(self._lambda)
            nn.init.zeros_(self._beta)

        if 'pos' in TEMPORAL_VARIATION:
            self.pos_enc = PositionalEncoding(IN_DIM)

        if 'tgat' in TEMPORAL_VARIATION:
            self.pos_enc = TGATEncoding(IN_DIM)

    def forward(self, graph, feat):
        if self._aggre_type != 'mean':
            raise KeyError(
                'Aggregator type {} not recognized.'.format(self._aggre_type))

        self._compatibility_check()
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            if 'att' in TEMPORAL_VARIATION:
                graph.edata['_a'] = th.exp(
                    -th.clamp(graph.edata['dt'].view(-1, 1) * self._lambda.view(1, -1) + self._beta, max=0))

            if 'pos' in TEMPORAL_VARIATION:
                graph.edata['_p'] = self.pos_enc(graph.edata['dt'])

            if 'tgat' in TEMPORAL_VARIATION:
                graph.edata['_p'] = self.pos_enc(graph.edata['dt'])

            msg_fn = fn.copy_src('h', 'm')
            if 'att' in TEMPORAL_VARIATION:
                if 'add' in TEMPORAL_VARIATION:
                    msg_fn = u_add_mul_e('h', '_p', '_a', 'm')
                elif 'cat' in TEMPORAL_VARIATION:
                    msg_fn = u_cat_mul_e('h', '_p', '_a', 'm')
                else:
                    msg_fn = fn.u_mul_e('h', '_a', 'm')
            else:
                if 'add' in TEMPORAL_VARIATION:
                    msg_fn = u_add_e('h', '_p', 'm')
                elif 'cat' in TEMPORAL_VARIATION:
                    msg_fn = u_cat_e('h', '_p', 'm')

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = th.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Message Passing
            graph.srcdata['h'] = feat_src
            graph.update_all(msg_fn, fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
            h_neigh = self.fc_neigh(h_neigh)

            rst = h_neigh

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst

# class DotScore(nn.Module):
#     def forward(self, edge_subgraph, x):
#         with edge_subgraph.local_scope():
#             edge_subgraph.ndata['x'] = x
#             edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'))
#             return edge_subgraph.edata['score'] 
        
class ComplexScore(nn.Module):
    def edge_func(self, edges):
        real_head, img_head = th.chunk(edges.src['emb'], 2, dim=-1)
        real_tail, img_tail = th.chunk(edges.dst['emb'], 2, dim=-1)
        real_rel, img_rel = th.chunk(edges.data['rel_emb'], 2, dim=-1)

        score = real_head * real_tail * real_rel \
                + img_head * img_tail * real_rel \
                + real_head * img_tail * img_rel \
                - img_head * real_tail * img_rel
        return {'score': th.sum(score, -1)}
    
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['emb'] = x
            edge_subgraph.apply_edges(lambda edges: self.edge_func(edges))
            return edge_subgraph.edata['score']
        
class DistMultScore(nn.Module):
    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['rel_emb']
        score = head * rel * tail
        return {'score': th.sum(score, dim=-1)}
    
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['emb'] = x
            edge_subgraph.apply_edges(lambda edges: self.edge_func(edges))
            return edge_subgraph.edata['score']
        
class TransEScore(nn.Module):
    def __init__(self, gamma=10, dist_ord='l2'):
        super(TransEScore, self).__init__()
        self.gamma = gamma
        self.dist_ord = dist_ord
        
    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['rel_emb']
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=self.dist_ord, dim=-1)}
    
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['emb'] = x
            edge_subgraph.apply_edges(lambda edges: self.edge_func(edges))
            return edge_subgraph.edata['score']
        
class RotatEScore(nn.Module):
    def __init__(self, gamma=12,emb_init=(12+2)/50):
        super(RotatEScore, self).__init__()
        self.gamma = gamma
        self.emb_init = emb_init
        
    def edge_func(self, edges):
        re_head, im_head = th.chunk(edges.src['emb'], 2, dim=-1)
        re_tail, im_tail = th.chunk(edges.dst['emb'], 2, dim=-1)

        phase_rel = edges.data['rel_emb'] / (self.emb_init / np.pi)
        re_rel, im_rel = th.cos(phase_rel), th.sin(phase_rel)
        re_score = re_head * re_rel - im_head * im_rel
        im_score = re_head * im_rel + im_head * re_rel
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        score = th.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        return {'score': self.gamma - score.sum(-1)}
    
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['emb'] = x
            edge_subgraph.apply_edges(lambda edges: self.edge_func(edges))
            return edge_subgraph.edata['score']

class GraphSAGE(nn.Module):
    def __init__(self, n_entities, in_features, out_features):
        super().__init__()
        if not PROVIDE_EMB:
            self.e_emb = nn.Embedding(n_entities, in_features)
        norm = None
        if NORM:
            norm = nn.LayerNorm(OUT_DIM, elementwise_affine=True)
        self.conv1 = CustomSAGEConv(in_features * (2 if 'cat' in TEMPORAL_VARIATION else 1), out_features,
                                    AGGREGATOR_TYPE, FEAT_DROP, True, norm, ACTIVATION)

    def forward(self, blocks):                                
        if not PROVIDE_EMB:
            x = self.e_emb(blocks[0].srcdata['_ID'])
            x = self.conv1(blocks[0], x)
        else:
            x = self.conv1(blocks[0], blocks[0].srcdata['e_emb'])
        return x


class LinkPredictionModel(nn.Module):
    def __init__(self, n_entities, in_features, out_features, score_func = "distmult"):
        super().__init__()
        self.r_emb = nn.Embedding(3, out_features)

        self.graph_model = GraphSAGE(n_entities, in_features, out_features)
        score_func = score_func.lower()
        if score_func=="complex":
            self.predictor = ComplexScore()
        elif score_func=="rotate":
            gamma = 12
            self.predictor = RotatEScore(gamma,emb_init=(gamma+2)/(out_features//2))
            self.r_emb = nn.Embedding(3, out_features//2)
        elif score_func=="transe":
            self.predictor = TransEScore()
        elif score_func=="distmult":
            self.predictor = DistMultScore()

    def forward(self, sub_graph, blocks):        
        x = self.graph_model(blocks)
        
        # positive
        sub_graph.edata["rel_emb"] = self.r_emb(sub_graph.edata["rel_type"])         
        pos_score = self.predictor(sub_graph, x)
        
        # negative
        # stupid way to generate negative sample (will improve it in the future)
        pos_rel_type = sub_graph.edata["rel_type"]
        neg_rel_type = th.tensor([0]*len(pos_rel_type)).to(device) 
        for i in range(len(pos_rel_type)):
            candidate_rel_type = random.choice([th.tensor(0),th.tensor(1),th.tensor(2)])
            while candidate_rel_type==pos_rel_type[i]:
                candidate_rel_type = random.choice([th.tensor(0),th.tensor(1),th.tensor(2)])
            neg_rel_type[i] = candidate_rel_type        
        sub_graph.edata["rel_emb"] = self.r_emb(neg_rel_type)
        neg_score = self.predictor(sub_graph, x)
        return pos_score, neg_score
    
    def refer(self, sub_graph, blocks):        
        x = self.graph_model(blocks)
        
        score_value = th.zeros(3, len(sub_graph.edata["rel_type"]))
        for i in range(3):
            sub_graph.edata["rel_emb"] = self.r_emb(th.tensor([i]*len(sub_graph.edata["rel_type"])).to(device)) 
            score_value[i] = self.predictor(sub_graph, x).view(-1) 
        return score_value.argmax(axis=0) # the larger, the better

def ce_loss(pos_score, neg_score):
    return F.cross_entropy(th.stack([pos_score, neg_score], dim=-1),
                           th.zeros(len(pos_score), device=pos_score.device).long())