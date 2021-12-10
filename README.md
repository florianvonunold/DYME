## DYME
#### A Dynamic Metric for Dialog Modeling Learned from Human Conversations
This repository contains the code and pre-trained models for the conference paper 'DYME: A Dynamic Metric for Dialog Modeling Learned from Human Conversations', which was published at the 28th International Conference on Neural Information Processing 2021 (ICONIP 2021). 
DYME is a BERT-based approach to model the change of utterance metrics within dialogs. 
The code in this repository allows you to train DYME, a neural network that predicts the utterance metrics of the next sentence for given a dialog.

![How DYME works](docs/DYME.gif)

## Repository structure
    
    ├── docs/               <- Slides of the conference presentation at ICONIP 2021
    │
    ├── src/                <- Source code for annotating dialogs, analyzing dialogs visually, and training DYME
    │    │   
    │    ├── models/                        <- DYME and Baseline models
    │    │   
    │    ├── analyze_dialog_metrics.ipynb   <- Jupyter Notebook for visualizing the progression of metrics in dialogs
    │    │   
    │    ├── annotate_dialogs.py            <- Automatically annotate dialogs in a given dataset with a set of metrics
    │    │   
    │    ├── configs.py                     <- Project configuration settings
    │    │   
    │    ├── dataset.py                     <- Torch dataset wrapper
    │    │   
    │    ├── metric_helpers.py              <- Helper functions for computing metrics
    │    │   
    │    ├── metrics.py                     <- Actual dialog metrics that are used to annotate dialogs 
    │    │   
    │    └── train_dyme.py                  <- Source code for training DYME
    │
    ├── Makefile            <- Useful commands for setting this project up
    │
    ├── README.md           <- The top-level README for developers using DYME: you are here =)
    │
    └── requirements.txt    <- Package requirements the python environment

## Setup
Clone the project, launch a terminal at the top-level directory and execute the following commands.
### 1. Create the virtual (conda) environment


Create the conda environment: `make create_environment`

Activate the environment: `conda activate dyme`

Install all package requirements: `make requirements`


### 2. Download the datasets
To download the DailyDialog and EmpatheticDialogues datasets into the `datasets/` folder run:

    make download_dailydialog_dataset  
    make download_empatheticdialogues_dataset

Note: The downloads may take a few moments to start.
You can also download the datasets manually. 
Make sure to extract *only the content* of each of the downloaded folders into `DYME/datasets/daildialog` and `DYME/datasets/empatheticdialogues`.

### 3. Preprocess the datasets
In this required preprocessing step, each utterance in every dialog in the given dataset is automatically annotated with a set of given metrics.
Start annotating the datasets by running:

    python annotate_dialogs.py --dataset dailydialog
    python annotate_dialogs.py --dataset empatheticdialogues

Running the script for a dataset will create a new subfolder under `preprocessed_datasets/` and store the preprocessing results in the following two files:

- `conversation_metrics.pickle`: the metric values for all utterances in all dialogs, stored by dialog length
- `conversation_metrics_raw_text.pickle`: the tokenized raw text of the dialogs in conversation_metrics.pickle

Note: You may want to run the script on a GPU to accelerate computation if you're using metrics that require neural networks to compute sentence embeddings (per default only metrics that don't require neural networks will be used).

#### Command line arguments
You can run annotate_dialogs.py with the following command-line arguments:

    --dateset DATASET_NAME (required; dataset must exist)
    --exclude_dataset_specific_labels (boolean flag to exclude the labels that are already existing in the dataset from the results; default: False --> include dataset specific labels)
    --metrics question utterance_length [...] (you can specify a custom subset of the metrics to compute; will use a default set of metrics otherwise)
    --custom_file_ending YOUR_CUSTOM_FILE_ENDING (you can specify a custom file ending)

List of metrics that work out-of-the-box (selected per default): 
 - question
 - conversation_repetition
 - self_repetition
 - utterance_repetition
 - word_repetition
 - utterance_length

List of metrics that require other models to be added manually to the project (refer to step 5 for further instructions):
 - infersent_coherence 
 - USE_similarity
 - word2vec_coherence 
 - deepmoji 
 - empathy 

### (Optional) 4. Analyze the preprocessed datasets
In your terminal, run `jupyter notebook` and navigate to `src/analyze_dialog_metrics.ipynb` in the browser window that opens up.

### (Optional) 5. Download and add the pre-trained models, vectors and weights for other metrics
You will need to download pre-trained models, vectors and weights if you wish to compute the following metrics:

- deepmoji sentiment and deepmoji coherence (torchmoji)
- infersent coherence
- USE similarity
- word2vec coherence
- empathy (emotional reactions, interpretations, explorations)

To get the pre-trained models, vectors and weights for all metrics listed above except for the empathy metrics please refer to the original repository https://github.com/natashamjaques/neural_chat. 
Use the same names and locations for the models, weights and vectors as in the original repository and add them to the top-level directory `DYME/`.
Additional information can also be found here: https://github.com/natashamjaques/neural_chat/tree/master/HierarchicalRL

For training the empathy classifiers that are required to compute the empathy metrics please refer to https://github.com/behavioral-data/Empathy-Mental-Health. 
The trained models should be stored in `DYME/empathy_mental_health/trained_models/{emotional_reactions|explorations|interpretations}.pth`.

## Train DYME
If you have completed all setup steps, you can simply run `python src/train_dyme.py` now.
With the default configuration, this will train `DYME` for `5 epochs`, using a `batch_size of 16`
and a `learning_rate of 5e-4`, on all dialogs of `length 2 to 9` in `DailyDialog` and `EmpatheticDialogues`.

## How to cite
If you use the code or models, please cite our paper:

von Unold F., Wintergerst M., Belzner L., Groh G. (2021) DYME: A Dynamic Metric for Dialog Modeling Learned from Human Conversations. In: Mantoro T., Lee M., Ayu M.A., Wong K.W., Hidayanto A.N. (eds) Neural Information Processing. ICONIP 2021. Communications in Computer and Information Science, vol 1516. Springer, Cham. https://doi.org/10.1007/978-3-030-92307-5_30

    @InProceedings{
        10.1007/978-3-030-92307-5_30,
        author="von Unold, Florian
        and Wintergerst, Monika
        and Belzner, Lenz
        and Groh, Georg",
        editor="Mantoro, Teddy
        and Lee, Minho
        and Ayu, Media Anugerah
        and Wong, Kok Wai
        and Hidayanto, Achmad Nizar",
        title="DYME: A Dynamic Metric for Dialog Modeling Learned from Human Conversations",
        booktitle="Neural Information Processing",
        year="2021",
        publisher="Springer International Publishing",
        address="Cham",
        pages="257--264",
        abstract="With increasing capabilities of dialog generation methods, modeling human conversation characteristics to steer the dialog generation towards natural, human-like interactions has garnered research interest. So far, dialogs have mostly been modeled with developer-defined, static metrics. This work shows that metrics change within individual conversations and differ between conversations, illustrating the need for flexible metrics to model human dialogs. We propose DYME, a DYnamic MEtric for dialog modeling learned from human conversational data with a neural-network-based approach. DYME outperforms a moving average baseline in predicting the metrics for the next utterance of a given conversation by about 20{\%}, demonstrating the ability of this new approach to model dynamic human communication characteristics. ",
        isbn="978-3-030-92307-5"
    }

## Contact information
For help or issues using DYME, please submit a GitHub issue.

For personal communication related to DYME, please contact [Florian von Unold](mailto:florian.von-unold@tum.de).
