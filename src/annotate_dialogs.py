import sys
import argparse
import metrics
import configs
import os
import pickle
import spacy
import csv
import numpy as np
from tqdm import tqdm


class DialogAnnotator:
    def __init__(self, dataset_name, metric_names, add_dataset_specific_labels):
        self.dataset_name = dataset_name
        self.metric_names = metric_names
        self.add_dataset_specific_labels = add_dataset_specific_labels

        self.conversations_by_length = {}  # holds the computed metric information for all conversations in the dataset
        self.conversations_by_length_raw_text = {}

        self.dataset_dir = configs.get_dataset_load_dir[self.dataset_name]
        self.results_filepath = configs.get_results_filepath[self.dataset_name]

        self.dataset_generator = self.get_dataset_generator()
        self.create_missing_directories()

        self.nlp = spacy.load("en_core_web_sm")

    def get_dataset_generator(self):
        if self.dataset_name == 'dailydialog':
            return self.dailydialog_generator()
        if self.dataset_name == 'empatheticdialogues':
            return self.empatheticdialog_generator()
        return None

    def dailydialog_generator(self):
        """
        A generator that can be iterated over (e.g. in a for loop), that iterates over the dailydialog dataset (4 files)
        :yield: a list of utterances (the current dialog) and the respective daily dialog labels per utterance.
        """
        # text and labels are stored in separate files in dailydialog dataset
        text_file = open(self.dataset_dir.joinpath('dialogues_text.txt'))
        topic_file = open(self.dataset_dir.joinpath('dialogues_topic.txt'))
        act_file = open(self.dataset_dir.joinpath('dialogues_act.txt'))
        emotion_file = open(self.dataset_dir.joinpath('dialogues_emotion.txt'))

        # iterate over all files simultaneously
        for text, topic, act, emotion in zip(text_file, topic_file, act_file, emotion_file):
            # remove newline, split utterances and remove last element (empty string produced by split)
            utterances = text.rstrip().split('__eou__')[:-1]
            conversation_length = len(utterances)
            topic_labels, act_labels, emotion_labels = self.convert_daily_dialog_labels(topic, act, emotion,
                                                                                        conversation_length)

            if len(act_labels) != conversation_length or len(emotion_labels) != conversation_length:
                # conversation may have multiple topic labels
                # Labels do not match conversation length for this conversation. Skipping the conversation
                continue

            yield utterances, (topic_labels, act_labels, emotion_labels) if self.add_dataset_specific_labels else False

    def empatheticdialog_generator(self):
        """
        A generator that can be iterated over (e.g. in a for loop), that iterates over the empatheticdialogs dataset
        :yield: a list of utterances (the current dialog)
        """
        # dialogs are stored in separate files (train, valid and test) in empatheticdialogues dataset
        for train_dev_test_filename in ['train.csv', 'valid.csv', 'test.csv']:
            with open(self.dataset_dir.joinpath(train_dev_test_filename), 'r') as csv_file:
                csv_file.readline()  # skip header line

                prev_conv_id = -1
                conversation = []
                for row in csv_file:  # every row contains an utterance of a conversation (conv_id,...,utterance)
                    row = row.split(',')
                    conv_id = row[0]
                    if conv_id != prev_conv_id and prev_conv_id != -1:  # new conversation starts
                        if len(conversation) > 1:
                            yield conversation, False  # yield conversation
                        # (note: don't yield dataset specific labels (False) <- no approved interpretation of selfeval)
                        conversation = []  # reset conversation (start collecting next conversation)

                    utterance = row[5]
                    utterance = utterance.replace('_comma_', ',')
                    conversation.append(utterance)
                    prev_conv_id = conv_id
                yield conversation, False  # yield last conversation

    def create_missing_directories(self):
        if not os.path.exists(configs.top_level_save_dir):
            print(f"Top level directory for preprocessed data missing. Creating the missing directory at {configs.top_level_save_dir} ...")
            os.makedirs(configs.top_level_save_dir)
            print('Directory successfully created.')
        if not os.path.exists(os.path.dirname(self.results_filepath)):
            print(f"Results directory for '{self.dataset_name}' missing. Creating the missing directory at {self.results_filepath} ...")
            os.makedirs(os.path.dirname(self.results_filepath))
            print('Directory successfully created.')

    def annotate_dialog_dataset(self):
        """
        Computes a bunch of metrics at each utterance position within dialogs of different lengths and stores the result
        as a dictionary - where the keys are conversation_lengths and the values are metric values for all conversations
        of the respective length.

        :return: nothing, but modifies the results dictionary 'conversations_by_length' which is later written to file

        The results dictionary 'conversations_by_length' has the form {conversation_length: conversation_metric_info}
        where conversation_metric_info is a 3d numpy array in the form [conversation, metric, utterances]
        Example:    if we select all conversations of length 4, (in this example we have 3 such conversations)
                    for which we computed two metrics each (e.g. sentiment & utterance length),
                    for the 4 utterances of the conversation,
                    we call conversations_by_length[4] and get:

                    conversations_by_length[4] = [[[2, 5, 4, 3],
                                                   [0.1, 0.4, 0.2, 0.3]],
                                                  [[4, 8, 2, 6],
                                                   [0.2, 0.2, 0.3, 0.2]],
                                                  [[7, 2, 3, 8],
                                                   [0.8, 0.5, 0.2, 0.7]]]
                    (3 conversations, 2 metrics, 4 utterances)
        """
        # loop over the dataset generator for the specified dataset (list of utterances and their labels (tuple))
        for utterances, dataset_specific_labels in tqdm(self.dataset_generator,
                                                        total=configs.dataset_size[self.dataset_name],
                                                        unit=' dialogs'):
            # convert all utterances in the dialog to lowercase and tokenize them with the spacy tokenizer
            utterances = [[token.text for token in self.nlp(utterance.lower()) if token.text != ' '] for utterance in utterances]

            conversation_length = len(utterances)

            # compute metrics for conversation
            metrics_for_conversation = self.compute_metrics_for_conversation(conversation=utterances)

            if dataset_specific_labels:  # if user specified add_dataset_specific_labels AND there are such spec. labels
                # combine computed metrics and dataset specific labels for the current conversation
                metrics_for_conversation = np.concatenate((metrics_for_conversation,
                                                           np.array(dataset_specific_labels)))

            # add metrics to results dict
            if conversation_length in self.conversations_by_length:  # there already are entries for the current length
                # append metrics to dict entry for current length
                self.conversations_by_length[conversation_length] = \
                    np.concatenate((self.conversations_by_length[conversation_length],
                                    np.array([metrics_for_conversation])))
                # add raw text of current dialog to raw text dict for current length
                self.conversations_by_length_raw_text[conversation_length].append(utterances)
            else:  # this is the first entry for the current length
                # set metric entry for current length
                self.conversations_by_length[conversation_length] = np.array([metrics_for_conversation])
                # add raw text of current dialog to raw text dict for current length
                self.conversations_by_length_raw_text[conversation_length] = [utterances]

        # store metric order in the result dictionary
        self.conversations_by_length['metric_order'] = self.metric_names

        if self.add_dataset_specific_labels:
            self.conversations_by_length['metric_order'] += self.get_order_of_dataset_specific_metrics()

        # the deepmoji metric function returns two metrics: 'sentiment' and 'coherence'
        # --> split single deepmoji entry into two entries 'deepmoji_sentiment' and deepmoji_coherence'
        if 'deepmoji' in self.conversations_by_length['metric_order']:  # only if deepmoji metric is active
            self.split_deepmoji_label()

        # the empathy metric function returns three metrics: emotional_reactions, interpretations and explorations
        # --> modify metric order: split single empathy entry into three entries
        if 'empathy' in self.conversations_by_length['metric_order']:  # only if empathy metric is active
            self.split_empathy_label()

    def get_order_of_dataset_specific_metrics(self):
        """
        Specifies the names and order of the dataset specific metrics.
        :return: list of the metric names in the right order
        """
        if self.dataset_name == 'dailydialog':
            return ['topic', 'act', 'emotion']
        return []

    def split_deepmoji_label(self):
        """
        Splits the single 'deepmoji' metric entry into two entries: 'deepmoji_sentiment' and deepmoji_coherence'
        :return: nothing, modifies the dictionary self.conversations_by_length
        """
        index_of_deepmoji_metric = self.conversations_by_length['metric_order'].index('deepmoji')
        self.conversations_by_length['metric_order'][index_of_deepmoji_metric] = 'deepmoji_sentiment'
        self.conversations_by_length['metric_order'] = \
            self.conversations_by_length['metric_order'][:index_of_deepmoji_metric + 1] + ['deepmoji_coherence'] + \
            self.conversations_by_length['metric_order'][index_of_deepmoji_metric + 1:]

    def split_empathy_label(self):
        """
        Splits the single empathy metric entry into three entries: emotional_reactions, interpretations and explorations
        :return: nothing, modifies the dictionary self.conversations_by_length
        """
        index_of_empathy_metric = self.conversations_by_length['metric_order'].index('empathy')
        self.conversations_by_length['metric_order'][index_of_empathy_metric] = 'emotional_reaction_level'
        self.conversations_by_length['metric_order'] = \
            self.conversations_by_length['metric_order'][:index_of_empathy_metric + 1] + [
                'interpretation_level'] + \
            self.conversations_by_length['metric_order'][index_of_empathy_metric + 1:]
        self.conversations_by_length['metric_order'] = \
            self.conversations_by_length['metric_order'][:index_of_empathy_metric + 2] + [
                'exploration_level'] + \
            self.conversations_by_length['metric_order'][index_of_empathy_metric + 2:]

    def compute_metrics_for_conversation(self, conversation):
        """Computes the specified metrics for a given conversation"""
        conversation_metrics = []

        for metric in self.metric_names:
            if metric not in configs.supported_metric_functions:
                raise NotImplementedError()
            metric_calculator = getattr(metrics, metric)  # get metric calculation function
            cur_metric_for_conv = metric_calculator(conversation)  # apply metric calcuation function
            if metric == 'deepmoji':  # save computation time by not computing the deepmoji embeddings twice
                conversation_metrics.append(cur_metric_for_conv[0])  # deepmoji sentiment
                conversation_metrics.append(cur_metric_for_conv[1])  # deepmoji coherence
            elif metric == 'empathy':
                conversation_metrics.append(cur_metric_for_conv[:, 0])  # emotional_reaction_level in the conversation
                conversation_metrics.append(cur_metric_for_conv[:, 1])  # interpretation_level in the conversation
                conversation_metrics.append(cur_metric_for_conv[:, 2])  # exploration_level in the conversation
            else:
                conversation_metrics.append(cur_metric_for_conv)

        return np.array(conversation_metrics)

    def convert_daily_dialog_labels(self, topic, acts, emotions, conversation_length):
        """
        Converts the daily dialog labels from lists of strings to lists of integers
        :return: converted topic, act and emotion labels (lists of integers)
        """
        topic = int(topic)  # convert topic (str) to int
        topic_labels = np.array([topic] * conversation_length)  # annotate each sentence with the dialog topic label
        acts = map(int, acts.split())  # convert acts (list of str) to ints
        act_labels = np.fromiter(acts, dtype=int)
        emotions = map(int, emotions.split())  # convert acts (list of str) to ints
        emotion_labels = np.fromiter(emotions, dtype=int)
        return topic_labels, act_labels, emotion_labels

    def write_results_to_pickle(self, custom_file_ending=None):
        """
        Store the results (dictionary with metrics and dictionary with raw text) in two separate pickle files under
        results/dataset_name/conversation_metrics.pickle and results/dataset_name/conversation_metrics_raw_text.pickle
        :param custom_file_ending: specify a custom file ending (optional)
        """
        if custom_file_ending is not None:
            self.results_filepath = "{0}_{2}{1}".format(*list(os.path.splitext(self.results_filepath)) + [custom_file_ending])

        # store dictionary with conversations (stored per dialog length) and metrics
        with open(self.results_filepath, 'wb') as handle:
            pickle.dump(self.conversations_by_length, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # store raw text dictionary
        raw_text_filepath = "{0}_{2}{1}".format(*list(os.path.splitext(self.results_filepath)) + ['raw_text'])
        with open(raw_text_filepath, 'wb') as handle:
            pickle.dump(self.conversations_by_length_raw_text, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='You can specify the following things: '
                    'dataset name, (required)'
                    'whether only empathy or all other metrics should be computed (default: all other metrics), '
                    'whether dataset specific labels should be excluded (default: False, include them), '
                    'a custom file ending (optional), '
                    'and a list of metrics to compute (optional).')
    parser.add_argument('--dataset', type=str, help='the name of the dialog dataset')
    parser.add_argument('--exclude_dataset_specific_labels', action='store_true',
                        help='exclude the labels that are already existing in the dataset to the results')
    parser.add_argument('--metrics', nargs='*', help='specify the desired metrics to compute')
    parser.add_argument('--custom_file_ending', type=str, help='add a custom ending to the result files')

    args = parser.parse_args()

    if not args.dataset:
        parser.error("please specify the name of a dialog dataset")
        sys.exit(0)

    if not args.metrics:  # if no metrics are specified by user, use the default metrics
        metric_names = ['question',
                        'conversation_repetition',
                        'self_repetition',
                        'utterance_repetition',
                        'word_repetition',
                        'utterance_length']
        # further optional metrics (require additional models to run on the input texts)
        # 'infersent_coherence',
        # 'USE_similarity',
        # 'word2vec_coherence',
        # 'deepmoji',
        # 'empathy'
    else:  # use custom subset of the metrics
        # see configs.py for all supported metrics (supported_metric_functions)
        metric_names = args.metrics

    print("Starting to compute the following metrics on the '%s' dataset: %s.\n"
          'Dataset specific labels will be %s.%s' %
          (args.dataset, str(metric_names), 'included' if not args.exclude_dataset_specific_labels else 'excluded',
           (" The resulting files will have the custom file ending: '%s'." % args.custom_file_ending)
           if args.custom_file_ending else ''))

    annotator = DialogAnnotator(dataset_name=args.dataset,
                                metric_names=metric_names,
                                add_dataset_specific_labels=(not args.exclude_dataset_specific_labels))
    annotator.annotate_dialog_dataset()
    annotator.write_results_to_pickle(custom_file_ending=(args.custom_file_ending if args.custom_file_ending
                                                          else None))
