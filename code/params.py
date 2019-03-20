
####PARAMETERS TO DEFINE
data_number = 2
num_iter = 500
batch_size = 200
corrupt_size = 10
slice_size = 3
regularization = 1e-4
save_per_iter = 100
learning_rate = 1
val_iter= 50
output_dir = ''
train_embeddings=True

###SPECIFIC PARAMETERS PER DATASET -> DO NOT CHANGE!
if data_number == 0:
    data_name = 'Wordnet'
    ontological_info=False
    embedding_size = 100
    fresh_entities=False

elif data_number==1:
    data_name = 'Freebase'
    ontological_info=False
    embedding_size = 100
    fresh_entities=False

elif data_number==2:
    data_name='Wordnet_Ont'
    ontological_info=True
    embedding_size = 100
    fresh_entities=False

elif data_number==3:
    data_name='Freebase_Ont'
    ontological_info=True
    embedding_size = 200
    fresh_entities=False

elif data_number==4:
    data_name='Wordnet_Fresh'
    ontological_info=True
    embedding_size = 100
    fresh_entities=True

elif data_number==5:
    data_name='Freebase_Fresh'
    ontological_info=True
    embedding_size = 200
    fresh_entities=True


data_path = '../data/'+data_name
output_path = '../output/'+data_name+'/'



