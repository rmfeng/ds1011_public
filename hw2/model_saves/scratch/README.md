
## Trial 1

I ran the basic training model for 1 iteration and wanted to demonstrate the save functionality

**Just showing off the MD functionality!**


Hyperparameters used:
 - num_epochs - 1
 - lr - 0.01
 - voc_size - 100000
 - train_loop_check_freq - 10
 - dropout_rnn - 0.5
 - dropout_fc - 0.5
 - batch_size - 4096
 - fc_hidden_size - 100
 - rnn_hidden_size - 50
 - cnn_hidden_size - 100
 - cnn_kernal_size - 3
 - rnn_num_layers - 1
 - check_early_stop - True
 - es_look_back - 50
 - no_imp_look_back - 25
 - decay_lr_no_improv - 0.5
 - es_req_prog - 0.0
 - optim_enc - <class 'torch.optim.adam.Adam'>
 - optim_dec - <class 'torch.optim.adam.Adam'>
 - scheduler - <class 'torch.optim.lr_scheduler.ExponentialLR'>
 - scheduler_gamma - 0.95
 - criterion - <class 'torch.nn.modules.loss.CrossEntropyLoss'>

Loader parameters used:
 - act_vocab_size - 100002
 - embedding_dim - 300
 - num_classes - 3


Control parameters used:
 - save_best_model - True
 - save_each_epoch - True
 - ignore_params - ['pre_trained_vecs']
 - snli_train_path - data/nli/snli_train.tsv
 - snli_val_path - data/nli/snli_val.tsv
 - mnli_train_path - data/nli/mnli_train.tsv
 - mnli_val_path - data/nli/mnli_val.tsv
 - pretrained_path - data/nli/wiki-news-300d-1M.vec
 - model_saves - model_saves/
 - model_path - model_saves/scratch/