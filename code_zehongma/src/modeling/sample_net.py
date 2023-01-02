import torch
import torch.nn as nn
class SampleNet(nn.Module):
    def __init__(self, n_length, max_num_frame):
        super(SampleNet, self).__init__()
        self.shallow_conv = nn.Conv2d(3,1,7,1,3)
        self.n_length = n_length
        self.max_num_frame = max_num_frame
        self.ratio = self.max_num_frame/self.n_length
        self.init_parameters()

    def forward(self, x):
        input_x = x
        h, w = x.size(-2), x.size(-1)
        x = x.view((-1, 3) + x.size()[-2:])
        x = self.shallow_conv(x)
        x = x.view(-1, self.n_length, x.size(-3), x.size(-2)*x.size(-1))
        for i in range(self.n_length-1):
            d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:], x[:,i+1,:,:]).unsqueeze(1)
            d = d_i if i == 0 else torch.cat((d, d_i), 1)
        PA = d.view(-1, 1*(self.n_length-1), h, w)
        
        batch_size = x.shape[0]
        diff_score = PA.sum(dim=(-2,-1))
        diff_score_pow = torch.pow(diff_score, self.ratio**0.5)

        # diff_score_pow = torch.pow(diff_score, 0.5)
        score_sum = diff_score_pow.sum(dim=-1, keepdim=True)
        score_fraction = torch.divide(diff_score_pow, score_sum)

        cumulative_mask = 1-torch.ones((self.n_length-1, self.n_length-1)).triu_(1).cuda()
        # import pdb
        # pdb.set_trace()
        cumulative_score = torch.einsum("bl,ml->bml", score_fraction, cumulative_mask).sum(dim=-1)
        interval = 1/(self.max_num_frame-1)
        sample_mat = torch.arange(self.max_num_frame)*interval

        # select frames from frame_0 to frame_(max_num_frame-1)
        selected_frames = None
        selected_indexs = None
        for i in range(self.max_num_frame):
            index = torch.argmin(torch.abs(cumulative_score - sample_mat[i]), dim=-1)
            selected_frames_i = input_x[torch.arange(batch_size), index].unsqueeze(dim=1)
            index_pad = index.unsqueeze(dim=1)
            if selected_frames is not None:
                selected_frames = torch.cat((selected_frames, selected_frames_i), dim=1)
                selected_indexs = torch.cat((selected_indexs, index_pad), dim=1)
            else:
                selected_frames = selected_frames_i
                selected_indexs = index_pad
        # import pdb
        # pdb.set_trace()
        return selected_frames
    
    def init_parameters(self):
        # state_dict = torch.load('./models/PAN/PAN_PA_something_resnet50_shift8_blockres_avg_segment8_e80.pth.tar')
        # for k, v in state_dict.items():
        #     conv_weight = v['module.PA.shallow_conv.weight']
        #     conv_bias = v['module.PA.shallow_conv.bias']
        # self.shallow_conv.weight = torch.nn.Parameter(conv_weight)
        # self.shallow_conv.bias = torch.nn.Parameter(conv_bias)
        nn.init.constant_(self.shallow_conv.weight, 0.001)
        nn.init.constant_(self.shallow_conv.bias, 0)
        self.shallow_conv.requires_grad = False
        # nn.init.normal_(self.shallow_conv.weight, 0, 0.001)
        # nn.init.constant_(self.shallow_conv.bias, 0)

