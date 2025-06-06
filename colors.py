from collections import namedtuple
from matplotlib.colors import ListedColormap
from matplotlib import colormaps as cm

purples = ['#46275a', '#4a295f', '#4d2c63', '#512e68', '#54316c', '#573370', '#5a3673', '#5d3877', '#603a7b', '#633d7e', '#663f82', '#694285', '#6b4488', '#6e468b', '#70498e', '#734b91', '#754d94', '#785097', '#7a529a', '#7d549c', '#7f579f', '#8159a2', '#835ba4', '#865da7', '#8860a9', '#8a62ab', '#8c64ae', '#8e67b0', '#9069b2', '#926bb4', '#946db6', '#9670b8', '#9872ba', '#9a74bc', '#9c76be', '#9e79c0', '#9f7bc2', '#a17dc4', '#a37fc5', '#a581c7', '#a784c9', '#a886ca', '#aa88cc', '#ac8ace', '#ad8ccf', '#af8fd1', '#b191d2', '#b393d4', '#b495d5', '#b697d6', '#b79ad8', '#b99cd9', '#bb9eda', '#bca0dc', '#bea2dd', '#bfa4de', '#c1a6df', '#c2a9e0', '#c4abe2', '#c5ade3', '#c7afe4', '#c8b1e5', '#cab3e6', '#cbb5e7', '#cdb7e8', '#cebae9', '#d0bcea', '#d1beeb', '#d3c0ec', '#d4c2ec', '#d6c4ed', '#d7c6ee', '#d9c8ef', '#dacaf0', '#dbccf0', '#ddcef1', '#ded0f2', '#e0d3f3', '#e1d5f3', '#e3d7f4', '#e4d9f5', '#e5dbf5', '#e7ddf6', '#e8dff7', '#eae1f7', '#ebe3f8', '#ece5f8', '#eee7f9', '#efe9f9', '#f1ebfa', '#f2edfb', '#f3effb', '#f5f1fb', '#f6f3fc', '#f8f5fc', '#f9f7fd', '#faf9fd', '#fcfbfe', '#fdfdfe', '#ffffff']

oranges = ['#642c00', '#692f00', '#6d3100', '#723400', '#763700', '#7a3900', '#7e3c00', '#823e00', '#864100', '#894300', '#8d4600', '#904800', '#934b00', '#974d00', '#9a5000', '#9d5200', '#a05400', '#a25700', '#a55900', '#a85b00', '#aa5e02', '#ad6004', '#af6205', '#b26507', '#b46709', '#b6690b', '#b96c0d', '#bb6e0f', '#bd7011', '#bf7213', '#c17515', '#c37717', '#c5791a', '#c77b1c', '#c87d1e', '#ca8021', '#cc8223', '#ce8425', '#cf8628', '#d1882a', '#d28a2d', '#d48d30', '#d58f32', '#d79135', '#d89338', '#d9953a', '#db973d', '#dc9940', '#dd9b43', '#de9e46', '#e0a049', '#e1a24c', '#e2a44f', '#e3a652', '#e4a855', '#e5aa58', '#e6ac5b', '#e7ae5e', '#e8b062', '#e9b265', '#eab468', '#ebb66b', '#ecb86f', '#edba72', '#edbc76', '#eebe79', '#efc07c', '#f0c280', '#f0c484', '#f1c687', '#f2c88b', '#f2ca8e', '#f3cc92', '#f4ce96', '#f4d099', '#f5d29d', '#f6d4a1', '#f6d6a5', '#f7d8a8', '#f7daac', '#f8dcb0', '#f8ddb4', '#f9dfb8', '#f9e1bc', '#f9e3c0', '#fae5c4', '#fae7c8', '#fbe9cc', '#fbebd0', '#fbecd4', '#fceed8', '#fcf0dc', '#fdf2e1', '#fdf4e5', '#fdf6e9', '#fdf7ed', '#fef9f2', '#fefbf6', '#fefdfa', '#ffffff']

# custom cmap
puor_custom = ListedColormap(purples + oranges[::-1])
or_custom = ListedColormap(oranges[::-1])

def _color_scheme():
    ColorScheme = namedtuple(
        'ColorScheme',
        'cmap_d cmap_s g p y'
    )

    colors = ColorScheme(
        puor_custom,
        or_custom,
        '#333333',
        '#A15FD3',
        '#F4AD15',
    )
    return colors
color_scheme = _color_scheme()

# def markers():
#     Markers = namedtuple('Markers', 'p, b, m')
#     markers = Markers('o', 'D', 'v')
#     return markers
