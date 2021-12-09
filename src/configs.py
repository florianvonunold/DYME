from pathlib import Path

# directories for raw dialog datasets and preprocessed datasets
project_dir = Path(__file__).resolve().parent.parent
top_level_data_dir = project_dir.joinpath('datasets')
top_level_save_dir = project_dir.joinpath('preprocessed_datasets')

# directories for trained models and cached data
models_save_dir = project_dir.joinpath('trained_models')
data_cache_dir = project_dir.joinpath('data_cache')

# dataset-specific directories for loading dialog data
get_dataset_load_dir = {'dailydialog': top_level_data_dir.joinpath('dailydialog'),
                        'empatheticdialogues': top_level_data_dir.joinpath('empatheticdialogues')}

# dataset-specific directories for saving annotated dialogs
get_results_filepath = {'dailydialog': top_level_save_dir.joinpath('dailydialog/conversation_metrics.pickle'),
                        'empatheticdialogues': top_level_save_dir.joinpath('empatheticdialogues/conversation_metrics.pickle')}

# needed for processing time estimation
dataset_size = {'dailydialog': 13118,
                'empatheticdialogues': 24847}

# metrics that can be used to annotate dialogs
supported_metric_functions = {'question',
                              'conversation_repetition',
                              'self_repetition',
                              'utterance_repetition',
                              'word_repetition',
                              'utterance_length',
                              'deepmoji',
                              'infersent_coherence',
                              'USE_similarity',
                              'word2vec_coherence',
                              'empathy'}

# mapping from names of categorical metrics to IDs
categorical_metrics = {
    'topic': {
        1: 'ordinary_life',
        2: 'school_life',
        3: 'culture_and_education',
        4: 'attitude_and_emotion',
        5: 'relationship',
        6: 'tourism',
        7: 'health',
        8: 'work',
        9: 'politics',
        10: 'finance'
    },
    'act': {
        1: 'inform',
        2: 'question',
        3: 'directive',
        4: 'commissive'
    },
    'emotion': {
        0: 'no_emotion',
        1: 'anger',
        2: 'disgust',
        3: 'fear',
        4: 'happiness',
        5: 'sadness',
        6: 'surprise'
    },
    'emotional_reaction_level': {
        0: 'no communication',
        1: 'weak communication',
        2: 'strong communication'
    },
    'interpretation_level': {
        0: 'no communication',
        1: 'weak communication',
        2: 'strong communication'
    },
    'exploration_level': {
        0: 'no communication',
        1: 'weak communication',
        2: 'strong communication'
    }
}
