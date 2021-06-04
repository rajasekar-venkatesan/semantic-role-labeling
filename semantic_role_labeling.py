"""
Semantic Role Labeling
"""

# Imports
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import pandas as pd


# Classes
class SRL:
    def __init__(self):
        self.model = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
        self.srl_args = [f'ARG{i}' for i in range(5)]
        self.srl_args.extend(['ARGM-TMP', 'ARGM-LOC', 'ARGM-DIR', 'ARGM-MNR', 'ARGM-PRP', 'ARGM-CAU', 'ARGM-REC', 'ARGM-ADV', 'ARGM-PRD'])
        self.srl_args.extend(['V', 'DESCRIPTION'])

    def get_predictions(self, sentence):
        result = self.model.predict(sentence=sentence)
        list_of_dicts = []
        words = result['words']
        for frame in result['verbs']:
            srl_dict = {key: [] for key in self.srl_args}
            desc = frame['description']
            verb = frame['verb']
            tags = frame['tags']
            srl_dict['DESCRIPTION'] = desc
            for tagid, tag in enumerate(tags):
                for srl_arg in self.srl_args:
                    if tag.endswith(f'-{srl_arg}'):
                        srl_dict[srl_arg].append(words[tagid])
            srl_dict = {k: " ".join(v).strip() for k, v in srl_dict.items() if k != 'DESCRIPTION'}
            srl_dict['DESCRIPTION'] = desc
            list_of_dicts.append(srl_dict)
        df = pd.DataFrame(list_of_dicts)
        df['SENTENCE'] = sentence
        return df, result

srl_model = SRL()