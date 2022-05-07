# CS6910-Assignment3
Assignment 3 of the CS6910: Fundamentals of Deep Learning course by Dipra Bhagat (CS21S048) and Subham Das (CS21S058)


## Project tree
 * [predictions](./predictions)
   * [predictions_attention.csv](./predictions/predictions_attention.csv)
   * [predictions_vanilla.csv](./predictions/[predictions_vanilla.csv)
 * [src](./src)
   * [gpt2-finetuning.ipynb](./src/gpt2-finetuning.ipynb)
   * [sequencetosequencemodel-Vanilla-wandb.ipynb](./src/sequencetosequencemodel-Vanilla-wandb.ipynb)
   * [sequencetosequencemodel-Vanilla.ipynb](./src/sequencetosequencemodel-Vanilla.ipynb)
   * [sequencetosequencemodel-with-attention-wandb.ipynb](./src/sequencetosequencemodel-with-attention-wandb.ipynb)
   * [sequencetosequencemodel-with-attention.ipynb](./src/sequencetosequencemodel-with-attention.ipynb)
 * [README.md](./README.md)

## Running instructions

* **All** the .ipynb notebooks are well structured and can be run cell by cell. (use **run all** option on jupyter/colab/kaggle. Or you can also run anually cell by cell)

* To train with **command line**, run the file ```train_cmd.py``` by passing the list of hyperparameters as command line arguments. Example below:

```python
py train_cmd -e 15 -act 'relu' -cell 'GRU' -lr 0.001 -ne 2 -nd 2 -hl 256 -d 0.3 -bs 256 -bw 3
```
The full form of the arguments are as follows:
```python
'''
-e: epochs 
-act: activaion function to use
-cell: cell type ('GRU', 'lstm' etc)
-lr: learning rate
-ne: number of encoders
-nd: number of decoders
-hl: hidden layer size
-d: dropout
-bs: batch size 
-bw: beam width(for beam decoder)
'''
```


## Run each file for appropriate purposes:

1. ```sequencetosequencemodel-Vanilla-wandb.ipynb```: Run this notebook to sweep over hyperparameters for the Vanilla model.

2. ```sequencetosequencemodel-Vanilla.ipynb```: Run this notebook to train and test with the best vanilla model.

3. ```sequencetosequencemodel-with-attention-wandb.ipynb```:Run this notebook to sweep over hyperparameters for the Attention model.

4. ```sequencetosequencemodel-with-attention.ipynb```: Run this notebook to train and test with the best Attention model.

5. ```gpt2-finetuning.ipynb```: Run this notebook to generate song lyrics starting with "I Love Deep Learning".

6. There are functions defined to build a custom encoder-decoder model and to prepare the raw text data generators for training and testing which need to be compiled.

7. The sweep config used in both vanilla and attention models are:

```python
sweep_config = {
    'method': 'bayes',            
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'epochs': {
            'values': [10, 15]
        },
        'batch_size': {
            'values': [128, 256]
        },
        'inp_emb_size': {
            'values': [256, 512]
        },
        'n_enc_layers': {
            'values': [1, 2]
        },
        'n_dec_layers': {
            'values': [3, 5]
        },
        'h_layer_size': {
            'values': [256, 512, 768]
        },
        'cell_type': {
            'values': ['RNN', 'LSTM', 'GRU']
        },
        'dropout' :{
            'values': [0, 0.3]
        },
        'beam_width': {
            'values': [1, 3, 5]
        },
        'learning_rate': {
            'values': [0.001, 0.0001, 0.0005]
        }
    }
}
```

8. The ```build_model()``` and ```train()```  functions has been made flexible with the following positional arguments which are initialized with the best parameters found by wandb sweep. Below is the example code snippet of the training of the attention based model:


```python
#Building the model with the best hyperparameters
model = build_model(len(input_char_dec), len(target_char_dec), 
                    inp_emb_size=512, n_enc_layers=2, 
                    n_dec_layers=5, h_layer_size=256, 
                    cell_type='LSTM', dropout=0.3, r_dropout=0.3)


#Training the model with best set of hyperparameters
model = train(model = model, train_input_data= [train_enc_x,train_dec_x], train_target_data= train_dec_y, 
                      val_input_data= [val_enc_x,val_dec_x], val_target_data= val_dec_y, beam_width= 5,
                      attention = True, batch_size= 128, optimizer = 'adam', learning_rate= 0.001, 
                      epochs= 15)
model.save("best_model_attn.h5")
```

9. To make predictions on the model, one can call the ```test()``` function with test data as argument. The predictions will be saved as  ```predictions_vanilla.csv``` file for the vanilla model and ```predictions_attention.csv``` file for the attention model. Example code snippet below:

```python
#Testing the model with best set of hyperparameters

test_acc, test_exact_K_acc, test_exact_acc, test_loss, test_true_out,\
test_pred_out, test_pred_scores,attn_scores, model = test(model, test_enc_x, 
                                                          test_dec_y, max_decoder_seq_length, 
                                                          target_char_enc, 
                                                          target_char_dec, test_x)
```

10. For visualizations use the following are the functions:
```python
#For visualizing sample predictions by the model
visualize_samples(test_x, test_true_out, test_pred_out, test_pred_scores)

#For plotting the best model
plot_model(model, to_file="model_attn.png", show_shapes=True)
```

11. For attention visualization use the following code:
```python
from __future__ import print_function
import ipywidgets as widgets
from ipywidgets import interact, Layout, IntSlider

'''One can directly use visualize_attention(sample_ind, decoder_index) to get the result if interaction isn't needed, 
but uncommenting below code is easier to use and offers a good way to choose the sample index and decoder index, and seeing the attention for K decoder predictons'''


@interact(sample_ind = IntSlider(min=0, max=len(test_x)-1, step=1, value=10, layout=Layout(width='800px')))
def f(sample_ind):
    print(f'Input : {test_x[sample_ind]}')
    print(f'Top {len(test_pred_out[sample_ind])} predictions : ')
    mx_len = 0
    for pred in test_pred_out[sample_ind]:
        print(pred)
        mx_len = max(mx_len, len(pred))
    
    @interact(character_ind = widgets.IntSlider(min=0, max=mx_len-1, step=1, value=0))
    def g(character_ind):
        visualize_attention(sample_ind, character_ind)
```

