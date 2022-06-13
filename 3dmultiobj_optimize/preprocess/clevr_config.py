

LABEL_MAP = ['cube',  'sphere',  'cylinder', 'unknown']

def get_category(name):
    if 'Cube' in name:
        return 0
    elif 'Sphere' in name:
        return 1
    elif 'Cylinder' in name:
        return 2
    else:
        return -1