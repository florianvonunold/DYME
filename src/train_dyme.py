import sys
import os
sys.path.append('..')  # append parent folder to make import configs work

import configs
from DYME.src.dataset import Dataset
from DYME.src.models.dyme import DYME
from DYME.src.models.baseline import Baseline

import neptune
import pickle
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup


def load_data(datasets, metrics_to_predict, dialog_lengths, max_prediction_position, tod_bert=False, multiple_samples_per_dialog=True):
    raw_texts = []
    labels = None
    features = None
    prediction_positions = None
    prediction_utterances = None
    arrays_initialized = False

    for dataset in datasets:
        # load texts and labels from pickle files
        metrics_filepath = configs.get_results_filepath[dataset]
        raw_text_filepath = "{0}_{2}{1}".format(*list(os.path.splitext(metrics_filepath)) + ['raw_text'])

        # get raw text of dialogs of the specified lengths
        with open(raw_text_filepath, 'rb') as raw_text_file:
            # get labels (metrics_to_predict) of utterances AT the specified position in dialogs of  specified lengths
            with open(metrics_filepath, 'rb') as metrics_file:
                all_dialogs = pickle.load(raw_text_file)
                all_metrics = pickle.load(metrics_file)

                metric_indices = [i for i, metric in enumerate(all_metrics['metric_order']) if metric in metrics_to_predict]

                for dialog_length in dialog_lengths:  # get all dialogs of the specified lengths
                    dialogs = all_dialogs[dialog_length]
                    # convert dialogs (arrays of arrays of tokens) to an array of utterance arrays (concat utterances)
                    if tod_bert:  # add special tokens: [SYS] speaker 1, [USR] speaker 2
                        dialogs = np.array([[' '.join([' %s ' % [' [SYS] ', ' [USR] '][id % 2]] + utterance) for id, utterance in enumerate(dialog)] for dialog in dialogs])
                        # note: CLS and SEP are added automatically
                    else:
                        dialogs = np.array([[' '.join(utterance) for utterance in dialog] for dialog in dialogs])

                    metrics = all_metrics[dialog_length]

                    if multiple_samples_per_dialog:  # predict at all possible positions of the current dialog
                        first_prediction_position = 1
                    else:  # predict only at the penultimate position of the current dialog  (data leakage control)
                        first_prediction_position = dialog_length-1

                    for predict_at_position in range(first_prediction_position, dialog_length):
                        # for all dialogs of the current length create a sample to predict at the current position

                        # raw_text is the model context: all utterances BEFORE the specified prediction_position
                        raw_texts += list(map(' '.join, dialogs[:, :predict_at_position]))

                        # labels are the specified metrics AT the specified position
                        cur_labels = metrics[:, metric_indices, predict_at_position]

                        # features are the specified metrics BEFORE the specified position
                        real_features = metrics[:, metric_indices, :predict_at_position]
                        # since we want to predict for earlier and later dialog positions with the same model
                        # we need use imputation ("fill up" missing values with -1) every time the prediction position
                        # is before the max_prediction_position
                        imputation = np.full((len(dialogs), len(metric_indices), max_prediction_position - predict_at_position), np.NaN)
                        # concatenate "real" features and imputation
                        cur_features = np.concatenate((real_features, imputation), axis=2)
                        cur_features = cur_features.reshape(-1, cur_features.shape[1] * cur_features.shape[2], order='F')
                        # append scalar 'prediction_position' as the very last feature
                        cur_features = np.c_[cur_features, np.repeat(predict_at_position, len(dialogs))]

                        cur_pred_positions = np.repeat(predict_at_position, len(dialogs))

                        # prediction_utterance is the text of the utterance AT the specified prediction_position
                        cur_pred_utterances = dialogs[:, predict_at_position]
                        # note: prediction_utterances is only for printing test samples at inference time!
                        #       it contains the raw text of the utterances for which we predict

                        if not arrays_initialized:  # initialize numpy arrays with first element
                            labels = cur_labels
                            features = cur_features
                            prediction_positions = cur_pred_positions
                            prediction_utterances = cur_pred_utterances
                            arrays_initialized = True
                        else:
                            labels = np.concatenate((labels, cur_labels), axis=0)
                            features = np.concatenate((features, cur_features), axis=0)
                            prediction_positions = np.concatenate((prediction_positions, cur_pred_positions), axis=0)
                            prediction_utterances = np.concatenate((prediction_utterances, cur_pred_utterances), axis=0)

    return raw_texts, labels, features, prediction_positions, prediction_utterances


def train_eval_loop(dataloaders, model, loss_function, metric_order, max_prediction_position, optimizer, scheduler, epochs, val_raw_texts, val_pred_utt_texts, neptune_logging, log_val_samples=True):
    print('Start training.')
    for epoch in tqdm(range(epochs)):
        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_train_loss_per_metric = None
        epoch_val_loss_per_metric = None
        epoch_train_loss_per_position = None
        epoch_val_loss_per_position = None

        for phase in ['train', 'val']:  # perform a train and validation phase in each epoch
            if phase == 'train':
                model.train()  # train mode
                torch.set_grad_enabled(True)  # enable gradient calculation
            else:
                model.eval()  # inference mode
                torch.set_grad_enabled(False)  # disable gradient calculation (saves computation)

            step = 0
            running_loss = 0
            running_loss_per_metric = torch.zeros(len(metric_order)).to(device)
            running_loss_per_position = torch.zeros(max_prediction_position).to(device)
            num_samples_per_position = torch.zeros(max_prediction_position).to(device)
            # ^ track how many samples need to be averaged

            for batch in tqdm(dataloaders[phase]):
                step += 1

                if model.model_type != 'baseline':  # bert
                    optimizer.zero_grad()  # zero the parameter gradients

                # forward pass
                logits = model(input_ids=batch['input_ids'].to(device),
                               attention_mask=batch['attention_mask'].to(device),
                               token_type_ids=batch['token_type_ids'].to(device),
                               numerical_features=batch['numerical_features'].to(device),
                               prediction_positions=batch['prediction_position'].to(device))

                # calculate loss
                loss = loss_function(logits, batch['labels'].to(device))

                if model.model_type != 'baseline':  # bert
                    if phase == 'train':  # backward & optimize only in training phase
                        loss.backward()

                        if neptune_logging:
                            # calculate gradient norms for logging
                            batch_grad_norm = (model.dense.weight.grad.norm().cpu().numpy() + model.out.weight.grad.norm().cpu().numpy()) / 2
                            neptune.log_metric('Batch Gradient Norm', batch_grad_norm)
                            neptune.log_metric('learning_rate', scheduler.get_last_lr()[0])

                        optimizer.step()
                        scheduler.step()

                # ---- logging -----
                batch_loss = loss.item()
                running_loss += batch_loss
                running_avg_loss = running_loss/step

                # calculate loss per metric (additional information)
                mse_per_metric = nn.MSELoss(reduction='none')
                batch_loss_per_metric = torch.mean(mse_per_metric(logits, batch['labels'].to(device)).detach(), dim=0)
                running_loss_per_metric += batch_loss_per_metric
                running_avg_loss_per_metric = running_loss_per_metric/step

                # calculate loss per prediction position (additional information)
                # 1. get loss per sample
                mse_per_sample = nn.MSELoss(reduction='none')
                loss_per_sample = torch.mean(mse_per_sample(logits, batch['labels'].to(device)).detach(), dim=1)
                prediction_position_per_sample = batch['prediction_position'].to(device)
                # 2. add loss at the corresponding prediction positions
                for idx, sample_loss in enumerate(loss_per_sample):
                    running_loss_per_position[prediction_position_per_sample[idx]-1] += sample_loss
                # 3. divide loss at the corresponding prediction positions by the number of samples in the batch that
                # had these positions
                unique_positions, counts = torch.unique(prediction_position_per_sample, return_counts=True)
                num_samples_per_position[unique_positions-1] += counts.to(device)
                running_avg_loss_per_position = running_loss_per_position/num_samples_per_position

                if neptune_logging:
                    log_neptune_metrics(phase, running_avg_loss, running_avg_loss_per_metric, metric_order,
                                        running_avg_loss_per_position, model_type=model.model_type)

                print(" - Running avg %s loss: %.4f, Per metric: %s [%s], Per position: %s"
                      % (phase, running_avg_loss, str(running_avg_loss_per_metric), ' '.join(metric_order), str(running_avg_loss_per_position)))

                if phase == 'train':
                    epoch_train_loss = running_avg_loss
                    epoch_train_loss_per_metric = running_avg_loss_per_metric
                    epoch_train_loss_per_position = running_avg_loss_per_position
                else:
                    epoch_val_loss = running_avg_loss
                    epoch_val_loss_per_metric = running_avg_loss_per_metric
                    epoch_val_loss_per_position = running_avg_loss_per_position

            print(" - Current %s epoch loss: %.4f, Per metric: %s, Per position: %s" %
                  (phase, running_avg_loss, str(running_avg_loss_per_metric), str(running_avg_loss_per_position)))

        print("\nEpoch train loss: %.4f, per metric: %s, per position: %s \nEpoch val loss: %.4f, per metric: %s, per position: %s" %
              (epoch_train_loss, epoch_train_loss_per_metric, epoch_train_loss_per_position, epoch_val_loss, epoch_val_loss_per_metric, epoch_val_loss_per_position))

        if neptune_logging:
            log_neptune_metrics('epoch_train', epoch_train_loss, epoch_train_loss_per_metric, metric_order,
                                epoch_train_loss_per_position, model_type=model.model_type)
            log_neptune_metrics('epoch_val', epoch_val_loss, epoch_val_loss_per_metric, metric_order,
                                epoch_val_loss_per_position, model_type=model.model_type)

        if log_val_samples:
            # print last sample of last (val) batch
            last_sample_id = batch['ids'][-1].item()
            sample_raw_text = val_raw_texts[last_sample_id]
            sample_pred_utt_text = val_pred_utt_texts[last_sample_id]
            prediction = logits[-1].detach().cpu().numpy()
            true_labels = batch['labels'][-1].cpu().numpy()

            text_to_log = '\nInput context: %s' \
                          '\nPredicting metrics for utterance: %s' \
                          '\nPrediction: %s' \
                          '\nTrue labels: %s' % (sample_raw_text, sample_pred_utt_text, str(prediction), str(true_labels))
            print(text_to_log)

            if neptune_logging:
                neptune.log_text('Val samples', text_to_log)

    print('Training finished!')


def log_neptune_metrics(phase, loss, loss_per_metric, metric_order, loss_per_position, model_type):
    metric_to_log = model_type + '_'
    metric_to_log += 'running_avg_%s' % phase if 'epoch' not in phase else phase

    # epoch/train/val loss
    neptune.log_metric('%s_loss' % metric_to_log, loss)

    # epoch/train/val loss per metric
    for metric_idx, metric_name in enumerate(metric_order):
        neptune.log_metric('%s_%s_loss' % (metric_to_log, metric_name), loss_per_metric[metric_idx].item())

    # epoch/train/val loss per position
    for position, loss in enumerate(loss_per_position):
        neptune.log_metric('%s_pos_%i_loss' % (metric_to_log, position+1), loss_per_position[position].item())


def assure_equal_num_samples(raw_texts, labels, features, prediction_positions, prediction_utterances):
    positions, num_samples = np.unique(prediction_positions, return_counts=True)
    max_num_samples = num_samples.min()
    equal_raw_texts = None
    equal_labels = None
    equal_features = None
    equal_pred_positions = None
    equal_pred_utterances = None
    first_position = positions[0]
    # take max_num_samples for every position
    for position in positions:
        indices_of_random_max_num_samples_for_position = np.random.choice(np.where(prediction_positions == position)[0],
                                                                        size=max_num_samples, replace=False)
        # should select first max_num_samples (sequential sampling) or random max_num_samples (random sampling, default)
        # indices_of_first_max_num_samples_for_position = np.where(prediction_positions == position)[0][:max_num_samples]
        # indices_of_random_max_num_samples_for_position = indices_of_first_max_num_samples_for_position
        cur_raw_texts = np.array(raw_texts)[indices_of_random_max_num_samples_for_position]
        cur_labels = labels[indices_of_random_max_num_samples_for_position, :]
        cur_features = features[indices_of_random_max_num_samples_for_position, :]
        cur_pred_positions = prediction_positions[indices_of_random_max_num_samples_for_position]
        cur_pred_utterances = prediction_utterances[indices_of_random_max_num_samples_for_position]
        if position == first_position:
            equal_raw_texts = cur_raw_texts
            equal_labels = cur_labels
            equal_features = cur_features
            equal_pred_positions = cur_pred_positions
            equal_pred_utterances = cur_pred_utterances
        else:
            equal_raw_texts = np.concatenate((equal_raw_texts, cur_raw_texts), axis=0)
            equal_labels = np.concatenate((equal_labels, cur_labels), axis=0)
            equal_features = np.concatenate((equal_features, cur_features), axis=0)
            equal_pred_positions = np.concatenate((equal_pred_positions, cur_pred_positions), axis=0)
            equal_pred_utterances = np.concatenate((equal_pred_utterances, cur_pred_utterances), axis=0)
    return list(equal_raw_texts), equal_labels, equal_features, equal_pred_positions, equal_pred_utterances


def create_missing_directories(models_dir, datacache_dir, should_save_model=False, should_cache_data=False):
    if should_save_model:
        if not os.path.exists(models_dir):
            print('Directory for saved models missing. Creating the missing directory ...')
            os.makedirs(models_dir)
            print('Successfully created directory: ' + str(models_dir))
    if should_cache_data:
        if not os.path.exists(datacache_dir):
            print('Directory for data cache missing. Creating the missing directory ...')
            os.makedirs(datacache_dir)
            print('Successfully created directory: ' + str(datacache_dir))


def cache_preprocessed_data(to_path, data_tuple):
    # store tuple of preprocessed data to pickle file
    print('Storing preprocessed data to ' + str(to_path) + '...')
    with open(to_path, 'wb') as handle:
        pickle.dump(data_tuple, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Successfully stored preprocessed data to cache dir!')


def load_cached_preprocessed_data(from_path):
    # load tuple of preprocessed data from pickle file
    print('Loading cached preprocessed data from ' + str(from_path) + '...')
    with open(from_path, 'rb') as handle:
        return pickle.load(handle)


def store_scalers(to_path, scalers_dict):
    # store tuple of preprocessed data to pickle file
    print('Storing scalers to ' + str(to_path) + '...')
    with open(to_path, 'wb') as handle:
        pickle.dump(scalers_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Successfully stored scalers to cache dir!')


def preprocess_data(use_cache, cache_data, cache_data_path):
    # check if there's already cached data for the current experiment
    if use_cache and os.path.exists(cache_data_path):  # load cached data
        raw_texts, labels, features, prediction_positions, prediction_utterances = load_cached_preprocessed_data(cache_data_path)
        print('Successfully loaded cached preprocessed data!')
    else:  # preprocess data from scratch
        print('Preprocessing data ...')
        raw_texts, labels, features, prediction_positions, prediction_utterances = \
            load_data(dataset_names, metrics_to_predict, dialog_lengths, max_prediction_position, tod_bert=use_tod_bert,
                      multiple_samples_per_dialog=create_multiple_samples_per_dialog)
        print('Preprocessing complete!')
        print('Data shape (samples, labels, features, positions, pred_utterances): ', len(raw_texts), labels.shape, features.shape, prediction_positions.shape,
              prediction_utterances.shape)
        if cache_data:  # cache the preprocessed data
            cache_preprocessed_data(cache_data_path, (raw_texts, labels, features, prediction_positions, prediction_utterances))
    return raw_texts, labels, features, prediction_positions, prediction_utterances


def normalize_and_impute_features(train_features, val_features, test_features):
    # normalize in each set separately (otherwise we would encode information from validation data and test data in
    # the training data)
    train_scaler = MinMaxScaler(feature_range=(0,1))
    val_scaler = MinMaxScaler(feature_range=(0, 1))
    test_scaler = MinMaxScaler(feature_range=(0, 1))

    train_features = train_scaler.fit_transform(train_features)  # ignores NaN (which we want for imputation!)
    val_features = val_scaler.fit_transform(val_features)
    test_features = test_scaler.fit_transform(test_features)

    # impute missing feature values (marked with NaN) with constant -1
    # note: since we want to predict for earlier and later dialog positions with the same model we need use imputation
    # ("fill up" missing values with -1) every time the prediction position is before the max_prediction_position
    train_features[np.isnan(train_features)] = -1
    val_features[np.isnan(val_features)] = -1
    test_features[np.isnan(test_features)] = -1

    return train_features, val_features, test_features, train_scaler


def normalize_labels(train_labels, val_labels, test_labels):
    train_label_scaler = MinMaxScaler(feature_range=(0, 1))
    val_label_scaler = MinMaxScaler(feature_range=(0, 1))
    test_label_scaler = MinMaxScaler(feature_range=(0, 1))

    train_labels = train_label_scaler.fit_transform(train_labels)
    val_labels = val_label_scaler.fit_transform(val_labels)
    test_labels = test_label_scaler.fit_transform(test_labels)
    return train_labels, val_labels, test_labels, train_label_scaler


def prepare_dataloaders(raw_texts, labels, features, prediction_positions, prediction_utterances, equal_num_samples_per_position):
    """
    Prepare dataloaders for fast dataloading during training/validation loop.
    """
    if equal_num_samples_per_position:
        raw_texts, labels, features, prediction_positions, prediction_utterances = \
            assure_equal_num_samples(raw_texts, labels, features, prediction_positions, prediction_utterances)

    # train / val / test split
    # split raw data into train & val (80% / 20%)
    train_texts, val_texts, train_labels, val_labels, train_features, val_features, \
    train_pred_positions, val_pred_positions, train_pred_utterances, val_pred_utterances = \
        train_test_split(raw_texts, labels, features, prediction_positions, prediction_utterances,
                         test_size=0.2, random_state=random_seed)
    # split validation data into val & test (50% / 50%)
    val_texts, test_texts, val_labels, test_labels, val_features, test_features, \
    val_pred_positions, test_pred_positions, val_pred_utterances, test_pred_utterances = \
        train_test_split(val_texts, val_labels, val_features, val_pred_positions, val_pred_utterances,
                         test_size=0.5, random_state=random_seed)

    # normalize each metric (feature) between 0 and 1
    train_features, val_features, test_features, train_scaler = normalize_and_impute_features(train_features, val_features, test_features)

    # normalize labels as well (smaller error terms, easier interpretation of MSE;
    #                           one larger scaled variable won't lift MSE anymore)
    train_labels, val_labels, test_labels, train_label_scaler = normalize_labels(train_labels, val_labels, test_labels)

    # store train_feature_scaler and train_label_scaler to pickle files to use the same scalers for inference later on
    store_scalers(scalers_save_path, {'feature_scaler': train_scaler, 'label_scaler': train_label_scaler})

    # all sequences are padded to the same length and are truncated to be no longer model’s maximum input length
    # note: CLS and SEP tokens are added automatically
    if use_tod_bert:  # TOD-BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('TODBERT/TOD-BERT-JNT-V1')
    else:  # BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # generate datasets (subclass of torch.utils.data.Dataset) for easy batch processing
    train_dataset = Dataset(train_encodings, train_labels, train_features, train_pred_positions)
    val_dataset = Dataset(val_encodings, val_labels, val_features, val_pred_positions)
    test_dataset = Dataset(test_encodings, test_labels, test_features, test_pred_positions)

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader, val_texts, val_pred_utterances


if __name__ == '__main__':
    # --------------------------------
    # HYPERPARAMETERS
    #  general
    experiment_suffix = 'default_metrics'  # will be appended to experiment identifier
    dataset_names = ['dailydialog', 'empatheticdialogues']  # datasets to use
    dialog_lengths = list(range(2, 9))  # dialogs of these lengths will be used (provide in ascending order, 17 is max!)
    create_multiple_samples_per_dialog = False
    #   True: predict at every dialog position of every single dialog (possible leakage of information across samples)
    #   False: only predict at penultimate position of each dialog (no leakage of information across samples)
    equal_num_samples_per_position = False  # use the same amount of samples per prediction position in the dialog
    use_numerical_features = True  # decide if metrics should be passed as input to the model or if just text is used
    enable_neptune_logging = False
    use_cache = False  # load data from cache (pickle file) if it exists
    cache_data = False  # save data to cache (pickle file)
    save_model = True
    random_seed = 7

    #  DYME
    use_tod_bert = False  # instead of BERT encoder, use TOD-BERT for dialog context encoding
    num_train_epochs = 5
    batch_size = 16
    learning_rate = 5e-4

    #  baseline
    train_baseline = True  # train baseline model on same data as the classifier (in the same run) for comparison
    baseline_model_type = 'mean'  # choose one of 'mean' and 'last' as the baseline type (default: 'mean')

    #  metrics
    metrics_to_predict = ['question',
                          'conversation_repetition',
                          'self_repetition',
                          'utterance_repetition',
                          'word_repetition',
                          'utterance_length']
    # further optional metrics (require additional models to run on the input texts during annotate_dialogs.py)
    # 'infersent_coherence'
    # 'USE_similarity'
    # 'word2vec_coherence'
    # 'deepmoji_sentiment'
    # 'deepmoji_coherence'
    # 'emotional_reaction_level'
    # 'interpretation_level'
    # 'exploration_level'

    # HYPERPARAMETERS END
    # --------------------------------

    # --------------------------------
    # NEPTUNE LOGGING
    if enable_neptune_logging:
        # Neptune setup for metric logging
        neptune.init(project_qualified_name='ADD_YOUR_NEPTUNE_PROJECT_HERE')  # initialize existing Neptune project
        experiment_name = f"DYME_{'_'.join(dataset_names)}_{dialog_lengths[0]}_{dialog_lengths[-1]}_{experiment_suffix}"
        neptune.create_experiment(name=experiment_name,
                                  params={'epochs': num_train_epochs, 'batch_size': batch_size,
                                          'learning_rate': learning_rate, 'seed': random_seed,
                                          'metrics': metrics_to_predict,
                                          'baseline_type': baseline_model_type if train_baseline else 'no baseline',
                                          'create_multiple_samples_per_dialog': create_multiple_samples_per_dialog})
    # NEPTUNE LOGGING END
    # --------------------------------

    # --------------------------------
    # MAIN START

    # create directories for model saving and data caching if desired
    model_save_dir = configs.models_save_dir
    data_cache_dir = configs.data_cache_dir
    create_missing_directories(model_save_dir, data_cache_dir, save_model, cache_data)

    # filenames for storing models and scalers, and caching data
    experiment_id = '%s_%i_%i_%s' % ('_'.join(dataset_names), dialog_lengths[0], dialog_lengths[-1], experiment_suffix)
    save_model_path = model_save_dir.joinpath(experiment_id)
    cache_data_path = data_cache_dir.joinpath(experiment_id + '.pickle')
    scalers_save_path = model_save_dir.joinpath(experiment_id + '_scalers.pickle')

    # use GPU if available
    if torch.cuda.is_available():
        print('Running on GPU.')
        device = torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # random seeds
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    dialog_lengths = sorted(dialog_lengths)  # ensure the last element is the maximum dialog length
    max_prediction_position = dialog_lengths[-1] - 1  # get the latest prediction position

    # load texts, labels and features (and prediction utterances for printing at inference) from pickle files
    raw_texts, labels, features, prediction_positions, prediction_utterances = preprocess_data(use_cache, cache_data, cache_data_path)

    print('Preparing data: train/val split, scaling, imputation and creating dataloaders...')
    train_dataloader, val_dataloader, test_dataloader, val_texts, val_pred_utterances = prepare_dataloaders(
        raw_texts, labels, features, prediction_positions, prediction_utterances, equal_num_samples_per_position)
    print('Data preparation complete!')

    # create model(s)
    print('Creating models...')
    # DYME
    models = [DYME(num_metrics=len(metrics_to_predict),
                   max_prediction_position=max_prediction_position,
                   include_numerical_features=use_numerical_features, tod_bert=use_tod_bert)]
    # BASELINE
    if train_baseline:
        models.append(Baseline(num_metrics=len(metrics_to_predict), baseline_type=baseline_model_type))
    print('Models successfully created!')

    # START TRAINING
    print('Start training.')
    for model in models:  # train all models on the exact same data
        torch.manual_seed(random_seed)  # we want to have the exact same random behavior for all models!

        model.to(device)
        model.train()

        optimizer = None
        scheduler = None

        if model.model_type != 'baseline':  # dyme or bert?
            # init optimizer and scheduler
            optimizer = AdamW(model.parameters(), lr=learning_rate)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=0,
                                                        num_training_steps=len(train_dataloader) * num_train_epochs)

        # init mean squared error loss
        mse_loss = nn.MSELoss()

        # start training
        dataloaders = {"train": train_dataloader, "val": val_dataloader}
        train_eval_loop(dataloaders, model, mse_loss, metrics_to_predict, max_prediction_position, optimizer, scheduler,
                        num_train_epochs, val_texts, val_pred_utterances, enable_neptune_logging, log_val_samples=True)

        if save_model and model.model_type != 'baseline':  # save dyme model
            print("Saving model...")
            torch.save(model.state_dict(), save_model_path)
            print("Saved model to: ", save_model_path)
