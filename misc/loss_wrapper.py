import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.ls = opt.label_smoothing
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, itera):
        out = {}
        if not sc_flag:
            c = 'LabelSmoothing' if self.ls > 0 else 'CrossEntropy'
            # print(f'----------sc_flag:{sc_flag}, crit:{c}------------')
            loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:])
        else:
            # print(f'----------sc_flag:{sc_flag}, crit:Reward------------')
            self.model.eval()
            with torch.no_grad(): # greedy search?
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')
            # print(f'===greedy_res:{greedy_res.shape}===')
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, 
                                                    opt={'sample_method':'sample'}, mode='sample')
            # print(f'===gen_result:{gen_result.shape}===')
            # print(f'===sample_logprobs:{sample_logprobs.shape}===')
            # print(f'===gts:{len(gts), gts}===')
            gts = [gts[_] for _ in gt_indices.tolist()] # ground truth samples each has 5 captions
            # print(f'===gts:{len(gts), gts}===') # gts : list
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt, itera)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
