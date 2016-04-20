############################################################################
# fileio.py
#   Contains classes and functions for reading and creating various types of
#     files
#   Originally created as part of the pywebify project but suitable for
#     reuse in other applications
############################################################################
__author__    = 'Steve Nicholes'
__copyright__ = 'Copyright (C) 2015 Steve Nicholes'
__license__   = 'GPLv3'
__version__   = '0.2'
__url__       = 'https://github.com/endangeredoxen/pywebify'


try:
    import configparser
except:
    import ConfigParser as configparser
import os
oswalk = os.walk
import pandas as pd
import pdb
import pathlib
import re
import sys
import textwrap
import ast
import win32clipboard
from xml.dom import minidom
from xml.etree import ElementTree
import numpy as np
from docutils import core
from natsort import natsorted
osjoin = os.path.join
st = pdb.set_trace


def convert_rst(file_name, stylesheet=None):
    """ Converts single rst files to html

    Adapted from Andrew Pinion's solution @
    http://halfcooked.com/blog/2010/06/01/generating-html-versions-of-
        restructuredtext-files/

    Args:
        file_name (str): name of rst file to convert to html
        stylesheet (str): optional path to a stylesheet

    Returns:
        None
    """

    settings_overrides=None
    if stylesheet is not None:
        if type(stylesheet) is not list:
            stylesheet = [stylesheet]
        settings_overrides = {'stylesheet_path':stylesheet}
    source = open(file_name, 'r')
    file_dest = os.path.splitext(file_name)[0] + '.html'
    destination = open(file_dest, 'w')
    core.publish_file(source=source, destination=destination,
                      writer_name='html',
                      settings_overrides=settings_overrides)
    source.close()
    destination.close()

    # Fix issue with spaces in figure path and links
    with open(file_name, 'r') as input:
        rst = input.readlines()

    with open(file_dest, 'r') as input:
        html = input.read()

    # Case of figures
    imgs = [f for f in rst if 'figure::' in f]

    for img in imgs:
        img = img.replace('.. figure:: ', '').replace('\n', '')
        if ' ' in img:
            img_ns = img.replace(' ','')
            idx = html.find(img_ns) - 5
            old = 'alt="%s" src="%s"' % (img_ns, img_ns)
            new = 'alt="%s" src="%s"' % (img, img)
            html = html[0:idx] + new + html[idx+len(old):]

            with open(file_dest, 'w') as output:
                output.write(html)

    # Case of substituted images
    imgs = [f for f in rst if 'image::' in f]

    for img in imgs:
        img = img.replace('.. figure:: ', '').replace('\n', '')
        if ' ' in img:
            img_idx = img.find('image:: ')
            img = img[img_idx+8:]
            img_ns = img.replace(' ','')
            idx = html.find(img_ns)
            html = html[0:idx] + img + html[idx+len(img_ns):]
            with open(file_dest, 'w') as output:
                output.write(html)

    # Case of links
    links = [f for f in rst if ">`_" in f]

    for link in links:
        link = re.search("<(.*)>`_", link).group(1)
        if ' ' in link:
            link_ns = link.replace(' ','')
            idx = html.find(link_ns)
            html = html[0:idx] + link + html[idx+len(link_ns):]


            with open(file_dest, 'w') as output:
                output.write(html)


def read_csv(file_name, **kwargs):
    """
    Wrapper for pandas.read_csv to deal with kwargs overload

    Args:
        file_name (str): filename
        **kwargs: valid keyword arguments for pd.read_csv

    Returns:
        pandas.DataFrame containing the csv data
    """

    # kwargs may contain values that are not valid in the read_csv function;
    #  we need to filter those out first before calling the function
    kw_master = ['filepath_or_buffer', 'sep', 'dialect', 'compression',
                 'doublequote', 'escapechar', 'quotechar', 'quoting',
                 'skipinitialspace', 'lineterminator', 'header', 'index_col',
                 'names', 'prefix', 'skiprows', 'skipfooter', 'skip_footer',
                 'na_values', 'true_values', 'false_values', 'delimiter',
                 'converters', 'dtype', 'usecols', 'engine',
                 'delim_whitespace', 'as_recarray', 'na_filter',
                 'compact_ints', 'use_unsigned', 'low_memory', 'buffer_lines',
                 'warn_bad_lines', 'error_bad_lines', 'keep_default_na',
                 'thousands', 'comment', 'decimal', 'parse_dates',
                 'keep_date_col', 'dayfirst', 'date_parser', 'memory_map',
                 'float_precision', 'nrows', 'iterator', 'chunksize',
                 'verbose', 'encoding', 'squeeze', 'mangle_dupe_cols',
                 'tupleize_cols', 'infer_datetime_format', 'skip_blank_lines']

    delkw = [f for f in kwargs.keys() if f not in kw_master]
    for kw in delkw:
        kwargs.pop(kw)

    return pd.read_csv(file_name, **kwargs)


def str_2_dtype(val, ignore_list=False):
    """
    Convert a string to the most appropriate data type
    Args:
        val (str): string value to convert
        ignore_list (bool):  ignore option to convert to list

    Returns:
        val with the interpreted data type
    """

    # Special chars
    chars = {'\\t':'\t', '\\n':'\n', '\\r':'\r'}

    # Remove comments
    v = val.split('#')
    if len(v) > 1:  # handle comments
        if v[0] == '':
            val = '#' + v[1].rstrip().lstrip()
        else:
            val = v[0].rstrip().lstrip()

    # Special
    if val in chars.keys():
        val = chars[val]
    # None
    if val == 'None':
        return None
    # bool
    if val == 'True':
        return True
    if val == 'False':
        return False
    # dict
    if ':' in val and '{' in val:
        val = val.replace('{','').replace('}','')
        val = re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', val)
        k = []
        v = []
        for t in val:
            k += [str_2_dtype(t.split(':')[0], ignore_list=True)]
            v += [str_2_dtype(t.split(':')[1])]
        return dict(zip(k,v))
    # tuple
    if val[0] == '(' and val[-1] == ')' and ',' in val:
        return ast.literal_eval(val)
    # list
    if (',' in val or val.lstrip(' ')[0] == '[') and not ignore_list \
            and val != ',':
        if val[0] == '"' and val[-1] == '"' and ', ' not in val:
            return str(val.replace('"', ''))
        if val.lstrip(' ')[0] == '[':
            val = val.lstrip('[').rstrip(']')
        val = val.replace(', ', ',')
        new = []
        val = re.split(',(?=(?:"[^"]*?(?: [^"]*)*))|,(?=[^",]+(?:,|$))', val)
        for v in val:
            if '=="' in v:
                new += [v.rstrip().lstrip()]
            elif '"' in v:
                new += [v.replace('"','').rstrip().lstrip()]
            else:
                new += [str_2_dtype(v.replace('"','').rstrip().lstrip())]
        if len(new) == 1:
            return new[0]
        return new
    # float and int

    try:
        int(val)
        return int(val)
    except:
        try:
            float(val)
            return float(val)
        except:
            v = val.split('#')
            if len(v) > 1:  # handle comments
                if v[0] == '':
                    return '#' + v[1].rstrip().lstrip()
                else:
                    return v[0].rstrip().lstrip()
            else:
                return val.rstrip().lstrip()


class ConfigFile():
    def __init__(self, path=None, paste=False):
        """
        Config file reader

        Reads and parses a config file of the .ini format.  Data types are
        interpreted using str_2_dtype and all parameters are stored in both
        a ConfigParser class and a multi-dimensional dictionary.  "#" is the
        comment character.

        Args:
            path (str): location of the ini file (default=None)
            paste (bool): allow pasting of a config file from the clipboard

        """

        self.config_path = path
        self.config = configparser.RawConfigParser()
        self.config_dict = {}
        self.is_valid = False
        self.paste = paste
        self.rel_path = os.path.dirname(__file__)

        if self.config_path:
            self.validate_file_path()
        if self.is_valid:
            self.read_file()
        elif self.paste:
            self.read_pasted()

        else:
            raise ValueError('Could not find a config.ini file at the '
                             'following location: %s' % self.config_path)

        self.make_dict()

    def make_dict(self):
        """
        Convert the configparser object into a dictionary for easier handling
        """
        self.config_dict = {s:{k:str_2_dtype(v)
                            for k,v in self.config.items(s)}
                            for s in self.config.sections()}

    def read_file(self):
        """
        Read the config file as using the parser option
        """

        self.config.read(self.config_path)

    def read_pasted(self):
        """
        Read from clipboard
        """
        win32clipboard.OpenClipboard()
        data = win32clipboard.GetClipboardData()
        win32clipboard.CloseClipboard()
        self.config.read_string(data)

    def validate_file_path(self):
        """
        Make sure there is a valid config file at the location specified by
        self.config_path
        """

        if os.path.exists(self.config_path):
            self.is_valid = True
        else:
            if os.path.exists(osjoin(self.rel_path, file)):
                self.config_path = osjoin(self.rel_path, self.config_path)
                self.is_valid = True


class Dir2HTML():
    def __init__(self, base_path, ext=None, **kwargs):
        """
        Directory to unordered html list (UL) conversion tool

        Args:
            base_path (str): top level directory or path to a list of files to
                             use in the UL
            ext (list): file extensions to include when building file list

        Keyword Args:
            build_rst (bool): convert rst files to html
            excludes (list): names of files to exclude from the UL
            from_file (bool): make the report from a text file containing a
                list of directories and files or just scan the
                base_path directory
            natsort (bool): use natural (human) sorting on the file list
            onclick (bool): enable click to open for files listed in the UL
            onmouseover (bool): enable onmouseover viewing for files listed in
                the UL
            rst_css (str): path to css file for rst files
            show_ext (bool): show/hide file extension in the file list

        Returns:

        """

        self.base_path = base_path
        self.build_rst = kwargs.get('build_rst', False)
        self.excludes = kwargs.get('excludes',[])
        self.files = []
        self.from_file = kwargs.get('from_file', False)
        self.natsort = kwargs.get('natsort', True)
        self.onclick = kwargs.get('onclick', None)
        self.onmouseover = kwargs.get('onmouseover', None)
        self.rst = ''
        self.rst_css = kwargs.get('rst_css', None)
        self.show_ext = kwargs.get('show_ext', False)

        self.ext = ext
        if self.ext is not None and type(self.ext) is not list:
            self.ext = self.ext.replace(' ','').split(',')
            self.ext = [f.lower() for f in self.ext]

        self.get_files(self.from_file)
        if self.build_rst:
            self.make_html()
        self.filter()
        self.files = self.files.drop_duplicates().reset_index(drop=True)
        self.nan_to_str()
        self.make_links()
        self.make_ul()

    def df_to_xml(self, df, parent_node=None, parent_name=''):
        """
        Builds an xml structure from a DataFrame

        Args:
            df (DataFrame):  directory structure
            parent_node (node|None):  parent node in the xml structure
            parent_name (str|''):  string name of the node

        Returns:
            node:  ElementTree xml representation of df
        """


        def node_for_value(name, value, parent_node, parent_name,
                           dir=False, set_id=None):
            """
            creates the <li><input><label>...</label></input></li> elements.
            returns the <li> element.
            """

            node= ElementTree.SubElement(parent_node, 'li')
            child= ElementTree.SubElement(node, 'A')
            if set_id is not None:
                child.set('id', set_id)
            if self.onmouseover and not dir:
                child.set('onmouseover', self.onmouseover+"('"+value+"')")
            if self.onclick and not dir:
                child.set('onclick', self.onclick+"('"+value+"')")
                child.set('href', 'javascript:void(0)')
            elif self.onclick and dir:
                child.set('onclick', self.onclick+"('"+value+"')")
                child.set('href', 'javascript:void(0)')
            child.text= name
            return node

        subdirs = [f for f in df.columns if 'subdir' in f]

        if parent_node is None:
            node = ElementTree.Element('ul')
        else:
            node = ElementTree.SubElement(parent_node, 'ul')
        node.set('id', 'collapse')

        if len(subdirs) > 0:
            groups = df.groupby(subdirs[0])
            for i, (n, g) in enumerate(groups):
                del g[subdirs[0]]
                if n == 'nan':
                    for row in range(0,len(g)):
                        node_for_value(g.filename.iloc[row],
                                       g.html_path.iloc[row], node,
                                       parent_name, set_id='image_link')
                else:
                    current_path_list = g.full_path.iloc[0].split(os.path.sep)
                    path_idx = current_path_list.index(n)
                    folder_path = \
                        os.path.sep.join(current_path_list[0:path_idx+1])
                    try:
                        folder_path = pathlib.Path(folder_path).as_uri()
                    except:
                        st()
                    child = node_for_value(n, folder_path, node,
                                           parent_name, dir=True)
                    self.df_to_xml(g, child, n)

        else:
            for row in range(0,len(df)):
                node_for_value(df.filename.iloc[row], df.html_path.iloc[row],
                               node, parent_name, set_id='image_link')

        return node

    def get_files(self, from_file):
        """
        Get the files for the report

        Args:
            from_file (bool):  use a text file to identify the directories
                and files to be used in the report

        """

        if from_file:
            # Build the list from a text file
            with open(self.base_path,'r') as input:
                files = input.readlines()
            temp = pd.DataFrame()
            files = [f.strip('\n') for f in files if len(f) > 0]
            for f in files:
                self.base_path = f
                self.get_files(False)
                temp = pd.concat([temp,self.files])
            self.files = temp.reset_index(drop=True)

        else:
            # Walk the base_path to identify all the files for the report
            self.files = []
            for dirName, subdirList, fileList in oswalk(self.base_path):
                if self.ext is not None:
                    fileList = [f for f in fileList
                                if f.split('.')[-1].lower() in self.ext]
                for fname in fileList:
                    temp = {}
                    temp['full_path'] = \
                        os.path.abspath(osjoin(self.base_path,dirName,fname))
                    temp['html_path'] = \
                        pathlib.Path(temp['full_path']).as_uri()
                    temp['ext'] = fname.split('.')[-1]
                    if self.from_file:
                        top = self.base_path.split(os.sep)[-1]
                        subdirs = temp['full_path']\
                                  .replace(self.base_path.replace(top,''),'')\
                                  .split(os.sep)
                    else:
                        subdirs = temp['full_path']\
                                  .replace(self.base_path+os.sep,'')\
                                  .split(os.sep)
                    temp['base_path'] = self.base_path
                    for i,s in enumerate(subdirs[:-1]):
                        temp['subdir%s' % i] = s
                    temp['filename_ext'] = subdirs[-1]
                    temp['filename'] = os.path.splitext(subdirs[-1])[0]
                    self.files += [temp]

            if len(self.files) == 0 and os.path.exists(self.base_path) \
                    and self.base_path.split('.')[-1] in self.ext:
                temp = {}
                temp['full_path'] = os.path.abspath(self.base_path)
                temp['html_path'] = pathlib.Path(temp['full_path']).as_uri()
                subdirs = temp['full_path'].split(os.sep)
                temp['base_path'] = os.sep.join(subdirs[0:-1])
                temp['filename'] = subdirs[-1]
                self.files += [temp]

            self.files = pd.DataFrame(self.files)

            # Sort the files
            if self.natsort:
                temp = self.files.set_index('full_path')
                self.files = \
                    temp.reindex(index=natsorted(temp.index)).reset_index()

    def filter(self):
        """
        Filter out any files on the exclude list
        """

        for ex in self.excludes:
            self.files = \
                self.files[~self.files.full_path.str.contains(ex, regex=False)]

        self.files = self.files.reset_index(drop=True)

    def make_html(self):
        """
        Build html files from rst files
        """

        self.rst = self.files[self.files.ext=='rst']
        idx_to_drop = []
        for i, f in self.rst.iterrows():
            convert_rst(f['full_path'], stylesheet=self.rst_css)
            self.files.iloc[i]['ext'] = 'html'
            self.files.iloc[i]['filename'] = \
                self.files.iloc[i]['filename'].replace('rst','html')
            self.files.iloc[i]['filename_ext'] = \
                self.files.iloc[i]['filename_ext'].replace('rst','html')
            self.files.iloc[i]['full_path'] = \
                self.files.iloc[i]['full_path'].replace('rst','html')
            self.files.iloc[i]['html_path'] = \
                self.files.iloc[i]['html_path'].replace('rst','html')

            # Check for same-named images
            for ext in [v for v in self.ext if v != 'html']:
                idx = self.files.query('full_path==r"%s"' %
                              self.files.iloc[i]['full_path']
                                       .replace('html',ext)) \
                                       .index
                if len(idx) > 0:
                    idx_to_drop += list(idx)

        self.files = self.files.drop(idx_to_drop).reset_index(drop=True)

    def make_links(self):
        """
        Build the HTML links
        """

        self.files['link'] = '''<A onmouseover="div_switch(' ''' + \
                             self.files.html_path.map(str) + \
                             '''')" onclick="HREF=window.open(' ''' + \
                             self.files.html_path.map(str) + \
                             '''')"href="javascript:void(0)">''' + \
                             self.files.filename.map(str) + \
                             '''</A><br>'''

    def make_ul(self):
        """
        Convert the DataFrame of paths and files to xml
        """

        element= self.df_to_xml(self.files)
        xml = ElementTree.tostring(element)
        xml = minidom.parseString(xml)
        self.ul = xml.toprettyxml(indent='  ')
        self.ul = self.ul.replace('<?xml version="1.0" ?>\n', '')

    def nan_to_str(self):
        """
        Replace NaN with a string version
        """

        self.files = self.files.replace(np.nan, 'nan')


class FileReader():
    def __init__(self, path, **kwargs):
        """
        Reads multiple raw data files into memory based on a partial path name
        or a list of files and populates them into a single pandas DataFrame
        or a list of DataFrames

        Args:
            path (str|list): partial path name or list of files

        Keyword Args:
            contains (str|list): search string(s) used to filter the file
                list; default=''
            concat (bool):  True=concatenate all DataFrames into one |
                False=return a list of DataFrames; default=True
            gui (bool):  True=use a PyQt4 gui prompt to select files |
                False=search directories automatically; default=False
            labels (list|str): adds a special label column to the DataFrame
                for distinguishing between files
                list=one entry per DataFrame added in order of self.file_list
                str=single label added to all files (ex. today's date,
                username, etc.)
            read (bool): read the DataFrames after compiling the file_list
            scan (bool): search subdirectories
            split_char (str|list): chars by which to split the filename
            split_values (list): values to extract from the filename based on
                file_split (ex. Filename='MyData_20151225_Wfr16.txt' -->
                file_split = '_' and split_values = [None, 'Date', 'Wafer']
            skip_initial_space (bool):  remove leading whitespace from
                split_values
            tag_char (str): split character for file tag values
                (ex. Filename='MyData_T=25C.txt' --> removes T= and adds 25C
                to a column named T
            verbose (bool): print file read progress

        """

        self.path = path
        self.contains = kwargs.get('contains', '')
        self.header = kwargs.get('header', True)
        self.concat = kwargs.get('concat', True)
        self.exclude = kwargs.get('exclude', [])
        self.ext = kwargs.get('ext', '')
        self.gui = kwargs.get('gui', False)
        self.labels = kwargs.get('labels', [])
        self.scan = kwargs.get('scan', False)
        self.read = kwargs.get('read', True)
        self.include_filename = kwargs.get('include_filename', True)
        self.split_char = kwargs.get('split_char', ['_'])
        self.split_values = kwargs.get('split_values', [])
        self.skip_initial_space = kwargs.get('skip_initial_space', True)
        self.tag_char = kwargs.get('tag_char', '=')
        self.file_df = None
        self.file_list = []
        self.verbose = kwargs.get('verbose', True)
        self.read_func = kwargs.get('read_func', read_csv)
        self.counter = kwargs.get('counter', True)
        self.kwargs = kwargs

        # Format the contains value
        if type(self.contains) is not list:
            self.contains = [self.contains]

        # Format ext
        if self.ext != '':
            if type(self.ext) is not list:
                self.ext = [self.ext]
            for i, ext in enumerate(self.ext):
                if ext[0] != '.':
                    self.ext[i] = '.' + ext

        # Overrides
        if type(self.split_char) is not list:
            self.split_char = list(self.split_char)
        if self.split_values is None:
            self.split_values = []

        if self.concat:
            self.df = pd.DataFrame()
        else:
            self.df = []

        self.get_files()

        if self.read:
            self.read_files()

    def get_files(self):
        """
        Search directories automatically or manually by gui for file paths to
        add to self.file_list
        """

        # Gui option
        if self.gui:
            self.gui_search()

        # If list of files is passed to FileReader with no scan option
        elif type(self.path) is list and self.scan != False:
            self.file_list = self.path

        # If list of files is passed to FileReader with a scan option
        elif type(self.path) is list and self.scan == True:
            for p in self.path:
                self.walk_dir(p)

        # If single path is passed to FileReader
        elif self.scan:
            self.walk_dir(self.path)

        # No scanning - use provided path
        else:
            self.file_list = [self.path]

        # Filter based on self.contains search string
        for c in self.contains:
            self.file_list = [f for f in self.file_list if c in f]

        # Filter out exclude
        for exc in self.exclude:
            self.file_list = [f for f in self.file_list if exc not in f]

        # Filter based on self.ext
        try:
            if self.ext != '':
                self.file_list = [f for f in self.file_list
                                  if os.path.splitext(f)[-1] in self.ext]
        except:
            raise ValueError('File name list is malformatted: \n   %s\nIf you '
                             'passed a path and ' % self.file_list + \
                             'meant to scan the directory, please set the '
                             '"scan" parameter to True')

        # Make a DataFrame of file paths and names
        self.file_df = pd.DataFrame({'path': self.file_list})
        if len(self.file_list) > 0:
            self.file_df['folder'] = \
                self.file_df.path.apply(
                        lambda x: os.sep.join(x.split(os.sep)[0:-1]))
            self.file_df['filename'] = \
                self.file_df.path.apply(lambda x: x.split(os.sep)[-1])
            self.file_df['ext'] = \
                self.file_df.filename.apply(lambda x: os.path.splitext(x)[-1])

    def gui_search(self):
        """
        Search for files using a PyQt4 gui
            Add new files to self.file_list
        """

        from PyQt4 import QtGui

        done = False
        while done != QtGui.QMessageBox.Yes:
            # Open the file dialog
            self.file_list += \
                QtGui.QFileDialog.getOpenFileNames(None,
                                                   'Pick files to open',
                                                   self.path)

            # Check if all files have been selected
            done = \
                QtGui.QMessageBox.question(None,
                                           'File search',
                                           'Finished adding files?',
                                           QtGui.QMessageBox.Yes |
                                           QtGui.QMessageBox.No,
                                           QtGui.QMessageBox.Yes)

        # Uniquify
        self.file_list = list(set(self.file_list))

    def parse_filename(self, filename, df):
        """
        Parse the filename to retrieve attributes for each file

        Args:
            filename (str): name of the file
            df (pandas.DataFrame): DataFrame containing the data found in
                filename

        Returns:
            updated DataFrame
        """

        filename = filename.split(os.path.sep)[-1]  # remove the directory
        filename = os.path.splitext(filename)[0] # remove the extension

        # Split tag values out of the filename as specified by split_values
        for i, sc in enumerate(self.split_char):
            if i == 0:
                file_splits = filename.split(sc)
            else:
                file_splits = [f.split(sc) for f in file_splits]

        if len(self.split_char) > 1:
            file_splits = [item for sublist in file_splits for item in sublist]

        # Remove initial whitespace
        if self.skip_initial_space:
            file_splits = [f.lstrip(' ') for f in file_splits]

        # Remove tag_char from split_values
        for i, fs in enumerate(file_splits):
            if self.tag_char is not None and self.tag_char in fs:
                file_splits[i] = file_splits[i].split(self.tag_char)[1]

        # file_splits = filename.split(self.file_split)
        for i, f in enumerate(self.split_values):
            if f is not None and i < len(file_splits):
                df[f] = str_2_dtype(file_splits[i], ignore_list=True)

        return df

    def read_files(self):
        """
        Read the files in self.file_list (assumes all files can be cast into
        pandas DataFrames)
        """

        for i, f in enumerate(self.file_list):

            # Read the raw data file
            try:
                if self.verbose:
                    print(textwrap.fill(f,
                                        initial_indent=' '*3,
                                        subsequent_indent=' '*6))
                elif self.counter:
                    # Print a file counter
                    previous = '[%s/%s]' % ((i), len(self.file_list))
                    counter = '[%s/%s]' % ((i+1), len(self.file_list))
                    bs = len(previous)
                    if i == 0:
                        bs = 0
                    if i < len(self.file_list) - 1:
                        print('\b'*bs + counter, end='')
                    if i == len(self.file_list) - 1:
                        print('\b'*bs, end='')
                    sys.stdout.flush()

                temp = self.read_func(f, **self.kwargs)

            except:
                raise ValueError('Could not read "%s".  Is it a valid data '
                                 'file?' % f)

            # Add optional info to the table
            if type(self.labels) is list and len(self.labels) > i:
                temp['Label'] = self.labels[i]
            elif self.labels == '#':
                temp['Label'] = i
            elif type(self.labels) is str:
                temp['Label'] = self.labels

            # Optionally parse the filename to add new columns to the table
            if hasattr(self, 'split_values') and type(self.split_values) is \
                    not list:
                self.split_values = [self.split_values]
            if len(self.split_values) > 0:
                temp = self.parse_filename(f, temp)

            # Add filename
            if self.include_filename:
                temp['Filename'] = f

            # Add to master
            if self.concat:
                self.df = pd.concat([self.df, temp])
            else:
                self.df += [temp]

    def walk_dir(self, path):
        """
        Walk through a directory and its subfolders to find file names

        Args:
            path (str): top level directory

        """

        for dir_name, subdir_list, file_list in oswalk(path):
            self.file_list += [os.path.join(dir_name, f) for f in file_list]