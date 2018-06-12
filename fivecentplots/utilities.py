import importlib
import os, sys
import pdb
import numpy as np
st = pdb.set_trace
# Get user default file
user_dir = os.path.expanduser('~')
default_path = os.path.join(user_dir, '.fivecentplots')
if os.path.exists(default_path) and default_path not in sys.path:
    sys.path = [default_path] + sys.path
    from defaults_dev import *


def get_current_values(df, text, key='@'):
    """
    Parse a string looking for text enclosed by 'key' and replace with the
    current value from the DataFrame

    Args:
        df (pd.DataFrame): DataFrame containing values we are looking for
        text (str): string to parse
        key (str): matching chars that enclose the value to replace from df

    Returns:
        updated string
    """

    if key in text:
        rr = re.search(r'\%s(.*)\%s' % (key, key), text)
        if rr is not None:
            val = rr.group(1)
            pos = rr.span()
            if val in df.columns:
                val = df[val].iloc[0]
            else:
                val = ''
            text = text[0:pos[0]] + str(val) + text[pos[1]:]

    return text


def get_decimals(value, max_places=10):
    """
    Get the number of decimal places of a float number excluding
    rounding errors

    Args:
        value (int|float): value to check
        max_places (int): maximum number of decimal places to check

    Returns:
        number of decimal places
    """

    last = np.nan
    for i in range(0, max_places+1):
        current = round(value, i)
        if current == last and current > 0:
            break
        else:
            last = current

    return i - 1


def kwget(dict1, dict2, val, default):
    """
    Augmented kwargs.get function

    Args:
        dict1 (dict): first dictionary to check for the value
        dict2 (dict): second dictionary to check for the value
        val (str): value to look for
        default (multiple): default value if not found in
            dict1 or dict2 keys

    Returns:
        value to use
    """

    if val in dict1.keys():
        return dict1[val]
    elif val in dict2.keys():
        return dict2[val]
    else:
        return default


def rectangle_overlap(r1, r2):
    """
    Determine if the bounds of two rectangles overlap

    Args:
        r1 (list|tuple): width, height, center point (x, y) of left rectangle
        r2 (list|tuple): width, height, center point (x, y) of right rectangle

    Returns:
        True if overlap | False if not
    """

    # Get bounds
    b1 = [r1[2][0] - r1[0]/2, r1[2][0] + r1[0]/2,
          r1[2][1] + r1[1]/2, r1[2][1] - r1[1]/2]
    b2 = [r2[2][0] - r2[0]/2, r2[2][0] + r2[0]/2,
          r2[2][1] + r2[1]/2, r2[2][1] - r2[1]/2]

    if b1[0]<b2[1] and b1[1]>b2[0] and b1[2]>b2[3] and b1[3]<b2[2]:
        return True
    else:
        return False


def reload_defaults():
    """
    Reload the fcp params
    """

    #del fcp_params
    import defaults
    importlib.reload(defaults)
    fcp_params = defaults.fcp_params
    colors = defaults.colors if hasattr(defaults, 'colors') else None
    markers = defaults.markers if hasattr(defaults, 'markers') else None

    return fcp_params, colors, markers


def set_save_filename(df, fig_item, fig_cols, layout, kwargs):
    """
    Build a filename based on the plot setup parameters

    Args:
        df (pd.DataFrame): input data used for the plot
        fig_item (None | str): unique figure item
        fig_cols (None | str | list): names of columns used to generate fi
        layout (obj): plot layout object
        kwargs (dict): input keyword args

    Returns:
        str filename
    """

    if 'filename' in kwargs.keys():
        return kwargs['filename'] + kwargs.get('save_ext', '.png')

    # Build a filename
    x = ' and '.join(validate_list(layout.label_x.text))
    y = ' and '.join(validate_list(layout.label_y.text))
    row, col, wrap, groups, fig = '', '', '', '', ''
    if layout.label_row.text is not None:
        row = ' by ' + layout.label_row.text
    if layout.label_col.text is not None:
        col = ' by ' + layout.label_col.text
    if layout.label_wrap.text is not None:
        wrap = ' by ' + layout.title_wrap.text
    # this one may need to change with box plot
    if kwargs.get('groups', False):
        groups = ' by ' + ' + '.join(validate_list(kwargs['groups']))
    if fig_item is not None:
        fig_items = validate_list(fig_item)
        for icol, col in enumerate(fig_cols):
            if icol > 0:
                fig += ' and'
            fig += ' where %s=%s' % (col, fig_items[icol])

    filename = '%s vs %s%s%s%s%s%s' % (x, y, row, col, wrap, groups, fig)

    return filename + kwargs.get('save_ext', '.png')


def validate_list(items):
    """
    Make sure a list variable is actually a list and not a single string

    Args:
        items (str|list): values to check dtype

    Return:
        items as a list
    """

    if items is None:
        return None
    if type(items) is tuple:
        return list(items)
    elif type(items) is not list:
        return [items]
    else:
        return items

