import numpy as np
import torch as t
import random
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
import torch.optim as optim
from external_funcs import *
import os
import datetime
from ntn_layer import NeuralTensorLayer
import loss_tensor
import params
import sys
import scipy.optimize as so
import pickle



def main():
    args=sys.argv
    if len(args) == 1:
        print 'Usage: python ntn.py {train/test} model_name'

    elif len(args) == 3 and args[1]=='train':
        print '-------------------- STARTING DATA PROCESSING --------------------'
        print '...............Loading entities and relations'
        entities_list=load_entities(params.data_path)
        relations_list=load_relations(params.data_path)
        num_entities = len(entities_list)
        num_relations = len(relations_list)
        entEmbed=None
        ont_info=None
        if not os.path.isdir(params.output_path):
            print '...............Creating output directory'
            os.mkdir(params.output_path)
        if not os.path.isfile(params.data_path+'/embedding_matrix.npy'):
            print '...............Starting creating embedding matrix'
            print '...............Calculating ent2word'
            init_word_embeds=load_obj(params.data_path+'/embeddings')
            ent2word=init_word_embeds.keys()
            print '...............Calculating word embeddings for each entity'
            entEmbed=np.array([init_word_embeds[entword] for entword in ent2word])
            print '...............Saving embedding matrix'
            np.save(params.data_path+'/embedding_matrix.npy',entEmbed)
        else:
            print '...............Loading embedding matrix'
            entEmbed=np.load(params.data_path+'/embedding_matrix.npy')
        print '...............Created embedding matrix'
        if params.ontological_info:
            if os.path.isfile(params.data_path+'/ont_embeddings.pkl'):
                ont_info=load_obj(params.data_path+'/ont_embeddings')
            else:
                print 'No ontological information provided!'
                exit()
        print '-------------------- PROCESSING TRAINING DATA --------------------'
        print '...............Loading training data'
        raw_training_data = load_training_data(params.data_path)
        print '...............Indexing training data'
        indexed_training_data = data_to_indexed(raw_training_data,entities_list, relations_list)

        print '-------------------- PROCESSING VALIDATION DATA --------------------'
        print '...............Loading validation data'
        dev_data = load_dev_data(params.data_path)
        print '...............Indexing validation data'
        indexed_dev_data= data_to_indexed(dev_data,entities_list,relations_list,val=True)

        print '-------------------- CREATING MODEL --------------------'
        if params.ontological_info:
            model=NeuralTensorLayer(params.embedding_size, params.slice_size, num_relations, entEmbed,ont_info=True,ont_matrix=ont_info,train_embeddings=params.train_embeddings)
        else:
            model = NeuralTensorLayer(params.embedding_size, params.slice_size, num_relations, entEmbed,train_embeddings=params.train_embeddings)
        optimizer=optim.LBFGS(model.parameters(),lr=params.learning_rate,max_iter=5)
        loss_criteria = loss_tensor.Tensor_Loss()
        print '...............Loss functions created'
        print '-------------------- STARTING TRAINING --------------------'
        for epoch in range(params.num_iter):
            training_batch=get_batch(params.batch_size, indexed_training_data, params.corrupt_size, option='train',fresh_entities=params.fresh_entities)

            if (params.save_per_iter != -1) and (epoch != 0) and (epoch % params.save_per_iter == 0):
                to_save = params.output_path+ args[2]+'_model_iter_' + str(epoch) + '.pt'
                t.save(model.state_dict(), to_save)
                validation_batch = get_batch(len(indexed_dev_data), indexed_dev_data, params.corrupt_size, option='val')
                score = model(validation_batch, eval=True)
                _, th = get_best_thresholds(score)
                np.save(params.output_path + '/thresholds_' + str(args[2]) + '_model_iter_'+ str(epoch)+'.npy', np.array(th))
                print '...............Model succesfully saved!'

            if (params.val_iter != -1) and (epoch != 0) and (epoch % params.val_iter == 0):
                validation_batch=get_batch(len(indexed_dev_data),indexed_dev_data,params.corrupt_size,option='val')
                score=model(validation_batch,eval=True)
                acc,th=get_best_thresholds(score)
                print '[ VALIDATION Iteration: ] Mean Accuracy -> '+str(np.mean(acc))
                for r in range(num_relations):
                    print '[ VALIDATION Iteration: ] Accuracy for relation '+str(r)+' with threshold '+str(th[r])+' -> '+str(acc[r])

            def closure():
                optimizer.zero_grad()
                score=model(training_batch)
                positive_score=score[0,:]
                negative_score=score[1,:]
                loss = loss_criteria([positive_score, negative_score], params.regularization, model.parameters())
                loss.backward()
                print '[ Iteration: ' + str(epoch) + ','+ str(
                    datetime.datetime.now()) + ' ] : Loss ' + str(np.array(loss.data))
                return loss

            optimizer.step(closure)

            if epoch == (params.num_iter - 1):
                validation_batch = get_batch(len(indexed_dev_data), indexed_dev_data,params.corrupt_size,option='val')
                score = model(validation_batch, eval=True)
                acc, th = get_best_thresholds(score)
                print '[ FINAL VALIDATION Iteration: ] Mean Accuracy -> ' + str(np.mean(acc))
                for r in range(num_relations):
                    print '[ FINAL VALIDATION Iteration: ] Accuracy for relation ' + str(r) + ' with threshold ' + str(
                        th[r]) + ' -> ' + str(acc[r])
                thresholds=np.array(th)
        to_save =params.output_path+ args[2]+'_model_final.pt'
        t.save(model.state_dict(), to_save)
        np.save(params.output_path + '/thresholds_' + args[2] + '_model_final.npy', thresholds)

    elif len(args) == 3 and args[1] == 'test':
        print '-------------------- STARTING DATA PROCESSING --------------------'
        print '...............Loading entities and relations'
        entities_list = load_entities(params.data_path)
        relations_list = load_relations(params.data_path)
        num_entities = len(entities_list)
        num_relations = len(relations_list)
        entEmbed = None
        if not os.path.isfile(params.data_path+'/embedding_matrix.npy'):
            print '...............Starting creating embedding matrix'
            print '...............Calculating ent2word'
            init_word_embeds=load_obj(params.data_path+'/embeddings')
            ent2word=init_word_embeds.keys()
            print '...............Calculating word embeddings for each entity'
            entEmbed=np.array([init_word_embeds[entword] for entword in ent2word])
            print '...............Saving embedding matrix'
            np.save(params.data_path+'/embedding_matrix.npy',entEmbed)
        else:
            print '...............Loading embedding matrix'
            entEmbed=np.load(params.data_path+'/embedding_matrix.npy')
        print '...............Created embedding matrix'
        if os.path.isfile(params.data_path + '/ont_embeddings.pkl') and params.ontological_info==True:
            ont_info = load_obj(params.data_path + '/ont_embeddings')
        print '-------------------- PROCESSING TEST DATA --------------------'
        print '...............Loading test data'
        test_data = load_test_data(params.data_path)
        print '...............Indexing test data'
        indexed_test_data = data_to_indexed(test_data, entities_list, relations_list, val=True)
        if params.fresh_entities:
            print '...............Loading fresh entities test data'
            fresh_test_data = load_fresh_test_data(params.data_path)
            print '...............Indexing fresh entities test data'
            fresh_indexed_test_data = data_to_indexed(fresh_test_data, entities_list, relations_list, val=True)
        print '-------------------- LOADING NTN MODEL --------------------'
        if params.ontological_info:
            model=NeuralTensorLayer(params.embedding_size, params.slice_size, num_relations, entEmbed,ont_info=True,ont_matrix=ont_info,train_embeddings=params.train_embeddings)
        else:
            model = NeuralTensorLayer(params.embedding_size, params.slice_size, num_relations, entEmbed,train_embeddings=params.train_embeddings)
        a=t.load(params.output_path+args[2]+'.pt')
        model.load_state_dict(a)
        # model=t.load(params.output_path+args[2]+'.pt')
        # th_name=str(args[2]).split('_')
        # th_name='_'.join(th_name[:th_name.index('model')])
        th_name=args[2]
        thresholds=np.load(params.output_path + '/thresholds_' + th_name + '.npy')
        print '-------------------- TESTING DATA --------------------'
        test_batch = get_batch(len(indexed_test_data), indexed_test_data, params.corrupt_size,
                                     option='val')
        total_score=[]
        values=model(test_batch,eval=True)
        r=0
        all_predictions = []
        all_labels = []
        for element in values:
            if element is None:
                print 'No metrics for relation ' + str(relations_list[r])
            else:
                scores = element[0, :]
                labels = element[1, :]
                scores = np.squeeze(np.array(scores))
                print '..............Evaluating triplets for relation ' + str(r)
                predictions = (scores >= thresholds[r]) * 2 - 1
                all_predictions = np.append(all_predictions, predictions)
                all_labels = np.append(all_labels, labels)
                labels= np.squeeze(np.array(labels))
                temp_accuracy = np.mean(predictions == labels)
                total_score.append(temp_accuracy)
                print 'Metrics for relation \"' + str(relations_list[r]) + '\": '
                get_metrics(predictions, labels)
                # print 'ACCURACY OF RELATION ' + str(r) + ' IS ' + str(temp_accuracy)
            r += 1
            print '..............................'
        print '-----------------------------------------------'
        # print 'TOTAL ACCURACY ' + str(np.mean(total_score))
        print 'TOTAL METRICS :'
        get_metrics(all_predictions, all_labels)

        if params.fresh_entities:
            print '-------------------- TESTING DATA WITH FRESH ENTITIES--------------------'
            test_batch = get_batch(len(fresh_indexed_test_data), fresh_indexed_test_data, params.corrupt_size,
                                   option='val')
            total_score = []
            values = model(test_batch, eval=True)
            r = 0
            all_predictions = []
            all_labels = []
            for element in values:
                if element is None:
                    print 'No metrics for relation ' + str(relations_list[r])
                else:
                    scores = element[0, :]
                    labels = element[1, :]
                    scores = np.squeeze(np.array(scores))
                    print '..............Evaluating triplets for relation ' + str(r)
                    predictions = (scores >= thresholds[r]) * 2 - 1
                    all_predictions = np.append(all_predictions, predictions)
                    all_labels = np.append(all_labels, labels)
                    labels = np.squeeze(np.array(labels))
                    temp_accuracy = np.mean(predictions == labels)
                    total_score.append(temp_accuracy)
                    print 'Metrics for relation \"' + str(relations_list[r]) + '\": '
                    get_metrics(predictions, labels)
                    # print 'ACCURACY OF RELATION ' + str(r) + ' IS ' + str(temp_accuracy)
                r += 1
                print '..............................'
            print '-----------------------------------------------'
            # print 'TOTAL ACCURACY ' + str(np.mean(total_score))
            print 'TOTAL METRICS :'
            get_metrics(all_predictions, all_labels)





if __name__=='__main__':
    main()