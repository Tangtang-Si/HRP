import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from layers.attention import Attention
from layers.csimdAtt import csimdAtt

classes = 2
epochs = 10
k1=0.4
k2=0.1
r=0.5


class Mgan(nn.Module):
    def __init__(self,inits):
        super(Mgan, self).__init__()
        self.fea_size = inits['fea_size']
        self.bert = BertModel.from_pretrained('./data/bert-base-uncased/')
        self.t_lin = nn.Linear(768, self.fea_size)
        local_model_path = "./data/vgg19/model.safetensors"
        self.vgg = timm.create_model('vgg19', pretrained=False, checkpoint_path=local_model_path)
        self.i_lin = nn.Linear(512, self.fea_size)
        self.LSTMAttention = csimdAtt(input_dim=self.fea_size, hidden_sizen=self.fea_size, output_dim=self.fea_size,dropout=0.5)
        self.rt_ri_att = Attention(embed_dim=self.fea_size, hidden_dim=self.fea_size, out_dim=self.fea_size,n_head=inits['num_head'], dropout=inits['drop'])
        self.maxpool=nn.MaxPool1d(kernel_size=3,stride=2)
        self.fc1 = nn.Linear(self.fea_size,2)
        self.fc0=nn.Linear(self.fea_size,self.fea_size)
        self.fc2=nn.Linear(4,self.fea_size)
        self.fc3=nn.Linear(self.fea_size*2,self.fea_size)
        self.fc4=nn.Linear(127,self.fea_size)
        self.fc5=nn.Linear(302,self.fea_size)
        self.fc6 = nn.Linear(self.fea_size * 4, self.fea_size)
        self.softplus = nn.Softplus()
        self.drop = nn.Dropout(inits['drop'])
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()
        self.t_ln = nn.LayerNorm(self.fea_size,elementwise_affine=True)
        self.i_ln = nn.LayerNorm(normalized_shape=[49, self.fea_size],elementwise_affine=True)

    def forward(self,inputs,epoch=0):
        label = inputs['label']
        last_hidden_state, _ = self.bert(input_ids = inputs['input_ids'],attention_mask = inputs['attention_mask'],return_dict=False)   #(4,302,768)# last_hidden_state.shape:[batch_size,squence_length,hidden_size=768]
        text1 = self.relu(self.t_lin(self.drop(last_hidden_state)))
        text2=text1.mean(1)
        text_c, text_s = self.LSTMAttention(text1)
        text= self.fc3(torch.cat([text_c, text_s],dim=1))

        image = self.vgg.forward_features(inputs['image_input']).reshape(-1,512,49).permute(0,2,1)
        image1 = self.relu(self.i_lin(self.drop(image)))
        image2=image1.mean(1)
        image_c, image_s = self.LSTMAttention(image1)
        image = self.fc3(torch.cat([image_c, image_s],dim=1))

        rt_ri, _ = self.rt_ri_att(image1,text1)
        rt_ri_t = self.t_ln((self.t_ln(rt_ri) + text1)).mean(1)
        rt_ri, _ = self.rt_ri_att(text1, image1)
        rt_ri_r = self.t_ln((self.t_ln(rt_ri) + image1)).mean(1)
        rt_ri = self.relu(self.fc6(torch.cat([rt_ri_t, rt_ri_r, text2, image2], dim=1)))

    # 三个单视图的分类结果
        text_mlp = self.fc1(self.fc0(self.relu(self.fc0(text))))
        text= self.softplus(text_mlp)
        rt_ri_mlp = self.fc1(self.fc0(self.relu(self.fc0(rt_ri))))
        rt_ri = self.softplus(rt_ri_mlp)
        image_mlp = self.fc1(self.fc0(self.relu(self.fc0(image))))
        image= self.softplus(image_mlp)

        evidence = [text, image, rt_ri]

        # 融合损失
        alpha = dict()
        loss = 0
        for v_num in range(len(evidence)):
            alpha[v_num] = evidence[v_num] + 1
            loss += ce_loss(label, alpha[v_num], classes, epoch)
        alpha_a= self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss(label, alpha_a, classes, epoch)
        fu_loss = torch.mean(loss)
        # 互补替代损失函数：r为互补和替代效应的协同因子，k2为互补替代损失的平衡因子
        comp = torch.norm(text_c - image_c, dim=1).pow(2).mean()
        subs = torch.norm(text_s - image_s, dim=1).pow(2).mean()
        CS_loss = -r * comp + (1 - r) * subs
        # 对比损失cl_loss：k1为对比损失的平衡因子
        extra_loss = fu_loss+k1*cl_loss(text,image)+k2*CS_loss
        return evidence_a, extra_loss

    def DS_Combin(self, alpha):

        def DS_Combin_two(alpha1, alpha2):

            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = classes / S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = classes / u_a
            # calculate new e
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a
def ce_loss(p, alpha, c, epoch):
    # p为标签，alpha为调整前的alpha,c为类别数，epoch为当前迭代次数
    S = torch.sum(alpha, dim=1, keepdim=True)
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    return A

def cl_loss(text, image):
    # k为调节对比损失的调节因子，一个样本为：评论，文本对
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = -cos(text, image)  # 同一样本的评论文本和评论图片的相似度
    loss = torch.sum(loss)
    loss += cl_dif(text, text)
    return loss

def cl_dif(x, y):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    xy = torch.sum(cos(x, y))  # 同一样本的评论文本和评论文本的相似度
    x = x.chunk(len(x), 0)
    y = y.chunk(len(x), 0)
    loss = 0
    for i in range(len(x)):
        for j in range(len(x)):
            loss += cos(x[i], y[j]) # 计算一个批次中所有的评论文本的相似度
    loss = torch.sum(loss - xy)
    return loss


