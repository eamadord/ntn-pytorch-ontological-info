# ntn-socher-pytorch
This repository contains a Pytorch-based implementation of the Neural Tensor Network (NTN) presented in [**Reasoning with Neural Tensor Networks for Knowledge Base Completion**  ](https://cs.stanford.edu/~danqi/papers/nips2013.pdf) (Richard Socher, Danqi Chen, Christopher D. Manning, Andrew Y. Ng, NIPS2013)

# Installation:
Clone the repository:

    git clone https://github.com/Elviish/ntn-socher-pytorch.git

Install the required packages:

    pip install -r requirements.txt

# Defining the parameters:
Enter the *code* directory:

    cd code

Modify the file *params.py* to specify the desired training parameters:

 -  *data number*: Select the ID of the desired dataset. The following table contains the specifics of each provided dataset:

| ID | Name | Embedding size | Ontological info | Fresh entities |
|----|----------------|----------------|------------------|----------------|
| **0** | Wordnet | 100 | No | No |
| **1** | Freebase | 100 | No | No |
| **2** | Wordnet_Ont | 100 | Yes | No |
| **3** | Freebase_Ont | 200 | Yes | No |
| **4** | Wordnet_Fresh | 100 | Yes | Yes |
| **5** | Freebase_Fresh | 200 | Yes | Yes |

 - *num_iter*: Number of iterations to run (default: 500)
 - *batch_size*: Number of samples per iteration
 - *corrupt_size*: Negative examples per positive examples
 - *slice_size*: Depth of the tensor
 - *regularization*: Value of the regularization param (default: 1e-4)
 - *save_per_iter*: Iterations between each saving point
 - *learning_rate*: Employed learning rate. As the employed optimization method is LBFGS, the default value is 1.
 - *val_iter*: Iterations between each validation pass.
 - *train_embeddings*: True if the embeddings of the entities are trained. False otherwise. (default: True)

# Training the model
To train the model, execute the following command:

    python ntn_ontology.py train {output_prefix}

The subsequent output generated models, as well as their associated thresholds,  are saved in **/output/{dataset}**.

# Testing the model
Once the model has been trained, we can test it. We can evaluate both the intermediate generated models and the final, with the command:

    python ntn_ontology.py test {model_name}

Metrics for both known entities and fresh entities test sets are provided for datasets *Wordnet_Fresh* and *Freebase_Fresh*.

