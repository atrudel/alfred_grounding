from grounding.models.base_models.clip import CLIPModelFrozen


class CLIPActionScorer:
    def __init__(self):
        self.clip: CLIPModelFrozen = CLIPModelFrozen

