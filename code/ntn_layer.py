import numpy as np
import torch as t
import torch.nn as nn
from torch.autograd import Variable
import external_funcs

class NeuralTensorLayer(nn.Module):
    def __init__(self,I_dim,S_dim,num_relations,embedding_matrix,ont_info=False,ont_matrix=None,train_embeddings=False):
        #1. Initialization of the hyperparameters of the layer
        self.output_dim=S_dim #Number of slices
        self.input_dim=I_dim #Dimension of the embedding vectors
        self.ont_info=False #Is there ontological information on the data?
        self.train_embeddings = train_embeddings #Are the embeddings of the entities going to be trained?
        self.relations = num_relations #Number of relations to predict
        if ont_info:
            self.ont_info=True
            self.input_dim=2*I_dim
            self.ont_matrix=ont_matrix

        #2. Initialization of the layer according to Pytorch
        super(NeuralTensorLayer, self).__init__()

        #3. Especification of the parameters of the layer
        self.W = nn.Parameter(nn.init.xavier_uniform_(t.zeros((self.input_dim,self.input_dim,self.output_dim,num_relations)))) #Matrix W: Dimension (DxDxKxR)
        self.V= nn.Parameter(nn.init.xavier_uniform_(t.zeros((self.output_dim,2*self.input_dim,num_relations)))) #Matrix V: Dimension (Kx2D,R)
        self.b= nn.Parameter(t.zeros((self.output_dim,1,num_relations))) #Vector B: Dimension (K,1,R)
        self.U= nn.Parameter(t.ones((1,self.output_dim,num_relations))) #Vector U: Dimension (1,K,R)
        #Matrix E (the training of this matrix is optional): Dimension (N_entities, D)
        if self.train_embeddings:
            self.E=nn.Parameter(t.Tensor(embedding_matrix))
        else:
            self.E=embedding_matrix

    def forward(self, batch,eval=False):
        relations_batch = external_funcs.split_batch(batch, self.relations) #Split the batch of triplets into R sub-batches (one for each relation) and index them by relation.
        predictions=[]
        k = self.output_dim
        if self.train_embeddings:
            entEmbed=t.Tensor.numpy(self.E.data)
        else:
            entEmbed=self.E
        for r in range(self.relations):
            training_data_batch=relations_batch[r] #Retrieve the batch corresponding to the current relation
            #TRAINING PASS:
            if not eval and training_data_batch!=[]:
                #Gather all indexes for the different entities
                e1 = [x[0] for x in training_data_batch] #e1 of both valid and corrupt triplets
                e2 = [x[1] for x in training_data_batch] #e2 of valid triplets
                e3 = [x[2] for x in training_data_batch] #e2 of corrupt triplets
                #Obtain the embeddings associated to the entities
                if self.ont_info:
                    e1_o_info=np.array([self.ont_matrix[x] for x in e1])
                    e2_o_info=np.array([self.ont_matrix[x] for x in e2])
                    e3_o_info=np.array([self.ont_matrix[x] for x in e3])
                    e1_input=Variable(t.Tensor(np.concatenate((e1_o_info,entEmbed[e1]),axis=1)))
                    e2_input = Variable(t.Tensor(np.concatenate((e2_o_info,entEmbed[e2]), axis=1)))
                    e3_input = Variable(t.Tensor(np.concatenate((e3_o_info,entEmbed[e3]), axis=1)))
                else:
                    e1_input = Variable(t.Tensor(entEmbed[e1]))
                    e2_input = Variable(t.Tensor(entEmbed[e2]))
                    e3_input = Variable(t.Tensor(entEmbed[e3]))
                #Secuence of operations to generate the scores for both positive and negative examples
                e1_pos=t.transpose(e1_input.data,0,1)
                e2_pos = t.transpose(e2_input.data, 0, 1)
                e1_neg = t.transpose(e1_input.data, 0, 1)
                e2_neg = t.transpose(e3_input.data, 0, 1)

                ff_product_pos=self.V.data[:,:,r].mm(t.cat([e1_pos,e2_pos],dim=0))
                ff_product_neg = self.V.data[:,:,r].mm(t.cat([e1_neg, e2_neg], dim=0))
                bilinear_products_pos=[]
                bilinear_products_neg=[]
                for i in range(k):
                    btp_pos=t.sum((t.transpose(e1_pos,0,1).mm(self.W.data[:,:,i,r])).mm(e2_pos),dim=0)
                    btp_pos=btp_pos.unsqueeze(0)
                    bilinear_products_pos.append(btp_pos)

                    btp_neg = t.sum((t.transpose(e1_neg, 0, 1).mm(self.W.data[:,:,i,r])).mm(e2_neg), dim=0)
                    btp_neg = btp_neg.unsqueeze(0)
                    bilinear_products_neg.append(btp_neg)
                concat_pos=t.cat(bilinear_products_pos,dim=0)
                concat_neg = t.cat(bilinear_products_neg, dim=0)
                pre_pos=concat_pos+ff_product_pos+self.b.data[:,:,r]
                pos_activation=t.tanh(pre_pos)
                pre_neg=concat_neg+ff_product_neg+self.b.data[:,:,r]
                neg_activation=t.tanh(pre_neg)
                score_pos=self.U.data[:,:,r].mm(pos_activation)
                score_neg=self.U.data[:,:,r].mm(neg_activation)
                predictions.append(t.cat((score_pos,score_neg),dim=0))
            elif eval:
            #VALIDATION / EVALUATION FORWARD PASS:
                if len(training_data_batch) == 0:
                    predictions.append(None)
                else:
                    #Gather indexes for all entities
                    e1 = [x[0] for x in training_data_batch]
                    e2 = [x[1] for x in training_data_batch]
                    #Gather the label for each triplet
                    labels = [x[2] for x in training_data_batch]
                    labels=t.Tensor(labels)
                    labels=labels.unsqueeze(0)
                    if self.ont_info:
                        e1_o_info = np.array([self.ont_matrix[x] for x in e1])
                        e2_o_info = np.array([self.ont_matrix[x] for x in e2])
                        e1_input = t.Tensor(np.concatenate((e1_o_info, entEmbed[e1]), axis=1))
                        e2_input = t.Tensor(np.concatenate((e2_o_info, entEmbed[e2]), axis=1))
                    else:
                        e1_input = t.Tensor(entEmbed[e1])
                        e2_input = t.Tensor(entEmbed[e2])

                    e1_input = t.transpose(e1_input, 0, 1)
                    e2_input = t.transpose(e2_input, 0, 1)
                    ff_product = self.V.data[:, :, r].mm(t.cat([e1_input, e2_input], dim=0))
                    bilinear_products = []
                    for i in range(k):
                        c = self.W.data[:, :, i, r]
                        b = t.transpose(e1_input, 0, 1).mm(c)
                        a = b.mm(e2_input)
                        btp = t.sum(a, dim=0)
                        btp = btp.unsqueeze(0)
                        bilinear_products.append(btp)
                    concat = t.cat(bilinear_products, dim=0)
                    pre_pos = concat + ff_product + self.b.data[:, :, r]
                    pos_activation = t.tanh(pre_pos)
                    score_pos = self.U.data[:, :, r].mm(pos_activation)
                    predictions.append(t.cat((score_pos, labels), dim=0))
        if not eval:
            #If we are training, we return a matrix of dimension (N_training examples, 2) containing the scores of the positive examples on the first column and
            #the scores of the negative examples on the second.
            predictions=t.cat(predictions,dim=1)
            return Variable(predictions)
        else:
            #If we are not training, we return a matrix of dimension (N_examples, 2) containing the score for each triplet and its label.
            return predictions






