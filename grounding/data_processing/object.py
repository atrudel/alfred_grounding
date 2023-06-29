from typing import Optional


class Object:
    def __init__(self, name: str):
        self.name: str = name.lower()
        self.index: int = object_names.index(self.name)
        self.templated_string_form: str = self.name.lower()  # Todo: change this
        # self.annotation_forms: List[str] = info['annotation_forms'].split(',')
        # self.substitutions: dict = {
        #     'sibling': info['sibling'],
        #     'generic': info['generic'],
        #     'description': info['description'],
        #     'unrelated': info['unrelated']
        # }

    def __str__(self) -> str:
        return self.name


# object_inventory = pd.read_csv('grounding/data_processing/objects.csv')
# object_list: List[Object] = [Object(row) for _, row in object_inventory.iterrows()]
object_names = [
    'alarmclock',
    'apple',
    'armchair',
    'baseballbat',
    'basketball',
    'bathtubbasin',
    'bed',
    'book',
    'bowl',
    'box',
    'bread',
    'butterknife',
    'cabinet',
    'candle',
    'cart',
    'cd',
    'cellphone',
    'cloth',
    'coffeemachine',
    'coffeetable',
    'countertop',
    'creditcard',
    'cup',
    'desk',
    'desklamp',
    'diningtable',
    'dishsponge',
    'drawer',
    'dresser',
    'egg',
    'floorlamp',
    'fork',
    'fridge',
    'garbagecan',
    'glassbottle',
    'handtowel',
    'handtowelholder',
    'kettle',
    'keychain',
    'knife',
    'ladle',
    'laptop',
    'lettuce',
    'microwave',
    'mug',
    'newspaper',
    'ottoman',
    'pan',
    'pen',
    'pencil',
    'peppershaker',
    'pillow',
    'plate',
    'plunger',
    'pot',
    'potato',
    'remotecontrol',
    'safe',
    'saltshaker',
    'shelf',
    'sidetable',
    'sinkbasin',
    'soapbar',
    'soapbottle',
    'sofa',
    'spatula',
    'spoon',
    'spraybottle',
    'statue',
    'stoveburner',
    'tennisracket',
    'tissuebox',
    'toilet',
    'toiletpaper',
    'toiletpaperhanger',
    'tomato',
    'vase',
    'watch',
    'wateringcan',
    'winebottle'
]

def bind_object(object_name: str) -> Optional[Object]:
    if object_name.lower() not in object_names:
        raise UnmatchedObjectException(f"{object_name} not in inventory")
    return Object(object_name)

class UnmatchedObjectException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"UnmatchedObjectException: {self.message}"