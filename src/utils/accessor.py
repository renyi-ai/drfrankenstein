from functools import reduce


def rgetattr(obj, attr, *args):
    ''' get attribute recursively, eg. self.layer0.conv.weight '''

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split('.'))
