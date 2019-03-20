import params
import scipy.io as sio
import numpy as np
import random
import pickle
import os


from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support


entities_string = '/entities.txt'
relations_string = '/relations.txt'
embeds_string = '/initEmbed.mat'
training_string = '/train.txt'
test_string = '/test.txt'
dev_string = '/dev.txt'
fresh_test_string='/test_fresh.txt'

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# input: path of dataset to be used
# output: python list of entities in dataset
def load_entities(data_path=params.data_path):
    entities_file = open(data_path + entities_string)
    entities_list = entities_file.read().strip().split('\n')
    entities_file.close()
    return entities_list


# input: path of dataset to be used
# output: python list of relations in dataset
def load_relations(data_path=params.data_path):
    relations_file = open(data_path + relations_string)
    relations_list = relations_file.read().strip().split('\n')
    relations_file.close()
    return relations_list


# input: path of dataset to be used
# output: python dict from entity string->1x100 vector embedding of entity as precalculated
def load_init_embeds(data_path=params.data_path):
    embeds_path = data_path + embeds_string
    return load_embeds(embeds_path)


# input: Generic function to load embeddings from a .mat file
def load_embeds(file_path):
    mat_contents = sio.loadmat(file_path)
    words = mat_contents['words']
    we = mat_contents['We']
    tree = mat_contents['tree']
    word_vecs = [[we[j][i] for j in range(params.embedding_size)] for i in range(len(words[0]))]
    entity_words = [map(int, tree[i][0][0][0][0][0]) for i in range(len(tree))]
    return (word_vecs, entity_words)


def load_training_data(data_path=params.data_path):
    training_file = open(data_path + training_string)
    training_data = [line.split('\t') for line in training_file.read().strip().split('\n')]
    return np.array(training_data)

###Bunch of positive and negative examples
def load_dev_data(data_path=params.data_path):
    dev_file = open(data_path + dev_string)
    dev_data = [line.split('\t') for line in dev_file.read().strip().split('\n')]
    return np.array(dev_data)


def load_test_data(data_path=params.data_path):
    test_file = open(data_path + test_string)
    test_data = [line.split('\t') for line in test_file.read().strip().split('\n')]
    return np.array(test_data)

def load_fresh_test_data(data_path=params.data_path):
    test_file = open(data_path + fresh_test_string)
    test_data = [line.split('\t') for line in test_file.read().strip().split('\n')]
    return np.array(test_data)

def data_to_indexed(data, entities, relations,val=False,fresh=False):
    entity_to_index = {entities[i]: i for i in range(len(entities))}
    relation_to_index = {relations[i]: i for i in range(len(relations))}
    if not val:
        indexed_data = [(entity_to_index[data[i][0]], relation_to_index[data[i][1]],\
                entity_to_index[data[i][2]]) for i in range(len(data))]
    else:
        if fresh:
            f = open(params.data_path + '/fresh_entities.txt')
            fresh_entities = f.read().strip().split('\n')
            f.close()

        indexed_data = [(entity_to_index[data[i][0]], relation_to_index[data[i][1]], \
                         entity_to_index[data[i][2]], int(data[i][3])) for i in range(len(data))]
    return indexed_data


def data_to_indexed_ontology(data,all_entities,selected_entities,relations,val=False):
    entity_to_index = {all_entities[i] : i for i in range(len(all_entities))}
    real_indexes={selected_entities[i] : entity_to_index[selected_entities[i]] for i in range(len(selected_entities))}
    relation_to_index = {relations[i]: i for i in range(len(relations))}
    if not val:
        indexed_data = [(real_indexes[data[i][0]], relation_to_index[data[i][1]], \
                         real_indexes[data[i][2]]) for i in range(len(data))]
    else:
        indexed_data = [(real_indexes[data[i][0]], relation_to_index[data[i][1]], \
                         real_indexes[data[i][2]],int(data[i][3])) for i in range(len(data))]
    return indexed_data

def get_batch(batch_size, data, corrupt_size,option='train',ont_info=False,fresh_entities=False):
    if batch_size>len(data):
        batch_size=len(data)
    all_entities = load_entities(params.data_path)
    # if ont_info and not fresh_entities and os.path.isfile(params.data_path+'/all_entities.txt'):
    #     f = open(params.data_path + '/all_entities.txt', 'r')
    #     all_entities_ont = [word.rstrip() for word in f]
    #     f.close()
    #     entity_to_index = {all_entities_ont[i]: i for i in range(len(all_entities_ont))}
    #     real_indexes = {all_entities[i]: entity_to_index[all_entities[i]] for i in
    #                     range(len(all_entities))}
    #     valid_indexes=real_indexes.values()
    if fresh_entities:
        f=open(params.data_path+'/fresh_entities.txt')
        fresh_entities=f.read().strip().split('\n')
        f.close()
        entity_to_index={all_entities[i]: i for i in range(len(all_entities))}
        fresh_indexes=[entity_to_index[fresh_entities[i]] for i in range(len(fresh_entities))]
        valid_indexes=list(set(range(len(all_entities))) - set(fresh_indexes))
        # valid_indexes = [x for x in range(len(all_entities)) if x not in fresh_indexes]
    else:
        valid_indexes=range(len(all_entities))
    random_indices = random.sample(range(len(data)), batch_size)
    if option is 'train':
        #data[i][0] = e1, data[i][1] = r, data[i][2] = e2, random=e3 (corrupted)
        batch = [(data[i][0], data[i][1], data[i][2], random.choice(valid_indexes))\
        for i in random_indices for _ in range(corrupt_size)]
    else:
        batch= [(data[i][0], data[i][1],data[i][2],data[i][3]) for i in range(batch_size)]
    return batch

def split_batch(data_batch, num_relations,option=None):
    batches = [[] for _ in range(num_relations)]
    if option=='fresh':
        for e1,r,e2 in data_batch:
            batches[r].append([e1,e2])
    else:
        for e1,r,e2,e3 in data_batch:
            batches[r].append((e1,e2,e3))
    return batches


###########################################3
def get_best_thresholds(results):
    best_accuracy=[-1 for _ in range(len(results))]
    best_threshold=[0.0 for _ in range(len(results))]
    interval = 0.001
    r=0
    for element in results:
        if element is None:
            best_threshold[r]=-1
            best_accuracy[r]=-1
        else:
            scores=element[0,:]
            labels=np.squeeze(np.array(element[1,:]))
            scores=np.squeeze(np.array(scores))
            score_min=np.min(np.array(scores))
            score_max=np.max(np.array(scores))
            score_temp=score_min

            while(score_temp <=score_max):
                predictions=(scores >= score_temp)*2-1
                temp_accuracy=np.mean(predictions==labels)

                if temp_accuracy > best_accuracy[r]:
                    best_threshold[r]=score_temp
                    best_accuracy[r]=temp_accuracy

                score_temp+=interval
        r+=1

    return best_accuracy,best_threshold


def get_metrics(predictions,labels):
    labels[labels==-1]=0
    predictions[predictions==-1]=0
    fpr, tpr, _ = roc_curve(labels, predictions)
    c_matrix=confusion_matrix(labels, predictions)
    precision,recall,_,_=precision_recall_fscore_support(labels,predictions,average='binary')
    tn,fp,fn,tp=c_matrix.ravel()
    print 'Accuracy -> '+str(np.mean(predictions==labels))
    print 'Recall -> '+str(recall)
    print 'Precision -> '+str(precision)
    print 'ROC Area -> '+str(auc(fpr,tpr))
    print 'TN Rate: '+str(tn)+' FP Rate: '+str(fp)+' FN Rate: '+str(fn)+' TP Rate: '+str(tp)
    print 'Confusion Matrix'
    print c_matrix


