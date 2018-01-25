# internal key match

def internal_key_match(key, keys):
    """
    internal rule for key matching between placeholders and data
    :param key:
    :param keys:
    :return: a bool and a data key
    """
    if key == 'advantage':
        if key in keys:
            return True, key
        elif 'return' in keys:
            return True, 'return'
        else:
            return False, None
    else:
        if key in keys:
            return True, key
        else:
            return False, None