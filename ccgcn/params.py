class Params:

    def __init__(self, 
                 ne=500, 
                 bsize=512, 
                 lr=0.001, 
                 reg_lambda=0.0, 
                 gcn_inp_dim=100,
                 gcn_hid_dim=150,
                 emb_dim=100, 
                 tim_dim=50,
                 neg_ratio=20, 
                 dropout=0.4,  
                 save_each=50,  
                 se_prop=0.9,
                 run_id='0',
                 lg_train_dir=None,
                 lg_valid_dir=None,
                 lg_test_dir=None,
                 time_dct=None,
                 inp_drop=0.1,
                 hid_drop=0.1,
                 gcn_drop=0.3,
                 fin_drop=0.4,
                 device='cpu',
                 is_gcn=1,
                 start_epoch=0,
                 opn='mult'
                 ):

        self.ne = ne
        self.run_id = run_id
        self.bsize = bsize
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.gcn_inp_dim = gcn_inp_dim
        self.gcn_hid_dim = gcn_hid_dim
        self.s_emb_dim = emb_dim
        self.t_emb_dim = tim_dim
        self.save_each = save_each
        self.neg_ratio = neg_ratio
        self.dropout = dropout
        self.se_prop = se_prop
        self.lg_train_dir = lg_train_dir
        self.lg_valid_dir = lg_valid_dir
        self.lg_test_dir = lg_test_dir
        self.time_dct = time_dct
        self.init_emb_dim = emb_dim
        self.inp_drop = inp_drop # input drop
        self.hid_drop = hid_drop # hidden drop
        self.gcn_drop = gcn_drop # gcn drop
        self.fin_drop = fin_drop # final drop
        self.is_gcn = bool(is_gcn)
        self.device = device
        self.start_epoch = start_epoch
        self.opn = opn
        
    def str_(self):
        return self.run_id 