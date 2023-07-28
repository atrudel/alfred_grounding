from grounding.models.frozen_models.clip import CLIPModel


class CLIPActionScorer:
    def __init__(self):
        self.clip: CLIPModel = CLIPModel

