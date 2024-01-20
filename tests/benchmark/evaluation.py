"""Function to process benchmark results in pandas."""
from pystencils.runhelper.db import Database, remove_constant_columns

db = None

def get_categorical(query):
    global db
    if db is None:
        db = Database('mongo://lbmpy_bench')
    res = basic_clean_up(db.to_pandas(query))
    res = make_categorical(res)
    return res


def get(query, **kwargs):
    global db
    if db is None:
        db = Database('mongo://lbmpy_bench')
    return basic_clean_up(db.to_pandas(query, **kwargs))


def remove_all_column_prefixes(df, inplace=False):
    """Strips everything left of a dots in pandas data frame column names: 'abc.def.value' is renamed to 'value'

    Similar to remove_prefix_in_column_name, that removes everything before the FIRST dot
    """
    if not inplace:
        df = df.copy()

    new_column_names = []
    for column_name in df.columns:
        if '.' in column_name:
            new_column_names.append(column_name[-column_name[::-1].index('.'):])
        else:
            new_column_names.append(column_name)
    df.columns = new_column_names
    return df


def basic_clean_up(df):
    """Cleans up a data frame that was loaded from the benchmark database.

    - fills default values for vectorization options
    - replaces columns that have stored lists with tuples
    - removes constant columns
    """
    if df is None or len(df) == 0:
        return df
    df = df.applymap(lambda e: tuple(e) if isinstance(e, list) else e)

    fill_default = {
        'optimization.vectorization.nontemporal': False,
        'optimization.vectorization.instruction_set': 'auto',
        'smagorinsky': False,
        'entropic': False,
        'cumulant': False,
        'stable': True,
    }
    categorical_columns = []  # ['optimization.vectorization.instruction_set']

    for col, default in fill_default.items():
        if col in df:
            df[col] = df[col].fillna(default)

    for col in categorical_columns:
        if col in df:
            df[col] = df[col].astype('category')

    df, constants = remove_constant_columns(df)
    remove_all_column_prefixes(df, inplace=True)
    return df


def make_categorical(df):
    """Summarizes boolean columns into categorical columns, such that plotting is simpler afterwards.

    - fixed_loop_sizes and fixed_relaxation_rates are summarized into single column with four values
    - same for 'cse_global' and 'cse_pdfs' columns
    """
    from pandas.api.types import CategoricalDtype

    def bool_to_category(col):
        df[col] = df.apply(lambda e: 'True' if e[col] else 'False', axis=1).astype('category')

    if all(c in df for c in ['instruction_set', 'assume_aligned', 'nontemporal']):
        def vec_column(row):
            if row['instruction_set'] == 'auto':
                return 'auto'
            else:
                result = str(row['instruction_set'])
                if row['assume_aligned']:
                    result += "-align"
                if row['nontemporal']:
                    result += "-nt"
                return result

        df['vec'] = df.apply(vec_column, axis=1)
        del df['instruction_set']
        del df['assume_aligned']
        del df['nontemporal']
        df['vec'] = df['vec'].astype('category')

    if all(c in df for c in ['method']):
        def method_category(row):
            method = row['method']
            if 'smagorinsky' in row and row['smagorinsky']:
                method += '-smag'
            if 'entropic' in row and row['entropic']:
                method += '-entr'
                if method.startswith('mrt3') and 'relaxation_rates' in row:
                    num_free_relaxation_rates = sum(1 for e in row['relaxation_rates'] if e == 'rr_free')
                    method += '-free{}'.format(num_free_relaxation_rates)
            if 'cumulant' in row and row['cumulant']:
                method += '-cumulant'
            return method
        df['method'] = df.apply(method_category, axis=1)
        for col in ['smagorinsky', 'entropic', 'cumulant', 'relaxation_rates']:
            if col in df:
                del df[col]

    if all(c in df for c in ['fixed_loop_sizes', 'fixed_relaxation_rates']):
        def fixed_column(row):
            mapping = {
                (False, False): 'generic',
                (True, False): 'loops only',
                (False, True): "ω's only",
                (True, True): "all fixed",
            }
            return mapping[(row['fixed_loop_sizes'], row['fixed_relaxation_rates'])]

        cat_type = CategoricalDtype(categories=['generic', "ω's only", "loops only", "all fixed"], ordered=True)
        df['fixed'] = df.apply(fixed_column, axis=1).astype(cat_type)
        del df['fixed_loop_sizes']
        del df['fixed_relaxation_rates']

    if 'fixed_loop_sizes' in df:
        bool_to_category('fixed_loop_sizes')
    if 'fixed_relaxation_rates' in df:
        bool_to_category('fixed_relaxation_rates')

    if all(c in df for c in ['cse_global', 'cse_pdfs']):
        def cse_column(row):
            mapping = {
                (False, False): 'none',
                (True, False): 'global only',
                (False, True): "pdfs only",
                (True, True): "full cse",
            }
            return mapping[(row['cse_global'], row['cse_pdfs'])]

        cat_type = CategoricalDtype(categories=["full cse", 'global only', 'none', "pdfs only", ], ordered=True)
        df['cse'] = df.apply(cse_column, axis=1).astype(cat_type)
        del df['cse_global']
        del df['cse_pdfs']

    if 'cse_global' in df:
        bool_to_category('cse_global')

    if 'cse_pdfs' in df:
        bool_to_category('cse_pdfs')

    if 'all_measurements' in df:
        del df['all_measurements']

    if 'split' in df:
        df['split'] = df.apply(lambda e: 'split' if e['split'] else 'no-split', axis=1).astype('category')

    if 'method' in df:
        df['method'] = df['method'].astype('category')

    return df


def speedup_table(df, column_name):
    import pandas as pd
    """Computes the speed up that a boolean optimization (column) causes."""
    result_columns = ['mlups_max', 'mlups_median', 'all_measurements']
    param_columns = list(set(df.columns) - set(result_columns + [column_name]))
    param_columns.insert(0, column_name)
    df = df.set_index(param_columns)
    result = pd.DataFrame(df.loc[True][['mlups_median']] / df.loc[False][['mlups_median']])
    return result.rename(columns={'mlups_median': 'speedup'})


def category_columns_to_string_columns(df):
    category_columns = []
    for c in df.columns:
        if df[c].dtype.name == 'category':
            category_columns.append(c)
            df[c] = df[c].astype(str)
    return category_columns

def speedup_table_categorical(df, column_name, slow_values):
    df = df.copy()
    df['_tmp_'] = df.apply(lambda row: False if row[column_name] in slow_values else True, axis=1)
    category_columns = category_columns_to_string_columns(df)
    del df[column_name]
    result = speedup_table(df, '_tmp_').reset_index()
    for c in category_columns:
        if c == column_name:
            continue
        result[c] = result[c].astype('category')
    return result


def flatten_index(df, keep=[], remove=[]):
    """See reset_index - pass index names to keep or to remove"""
    flatten_indices = []
    if remove:
        assert not keep
        for i, cn in enumerate(df.index.names):
            if cn in remove:
                flatten_indices.append(i)
    else:
        for i, cn in enumerate(df.index.names):
            if cn not in keep:
                flatten_indices.append(i)
    return df.reset_index(level=flatten_indices)


def bokeh_scatter_plot(df, category_column, dof_column, color_column=None, enable_hover=True, plot_size=(400, 300),
                       source=None, log=False):
    """Interactive bokeh scatter plot.

    Args:
        df: pandas data frame with data
        category_column: column name for data that is plotted on y axis (has to be categorical column)
        dof_column: column name plotted on the x axis (numeric)
        color_column: categorical column used to color the data points
        enable_hover: switch for tooltips on hover
        plot_size: (width, height) of plot
        source: use this parameter to link multiple bokeh plots together, pass the source here that was returned
                by the first plot, make sure to use the same data frame for all plots

    Returns:
        (plot, source to pass to next plot to link them together)
    """
    from bokeh.plotting import figure, ColumnDataSource
    from bokeh.models import HoverTool, WheelZoomTool
    from bokeh.transform import jitter, factor_cmap
    from bokeh.palettes import d3

    if source is None:
        source = ColumnDataSource(df)

    if df[category_column].dtype.name == 'category':
        figure_kwargs = {'y_range': [str(e) for e in df[category_column].unique()]}
    else:
        figure_kwargs = {}

    if log:
        figure_kwargs['x_axis_type'] = 'log'

    p = figure(plot_width=plot_size[0], plot_height=plot_size[1],
               tools="reset,pan,box_select,wheel_zoom", toolbar_location="right", **figure_kwargs)
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)

    kwargs = {}
    if color_column:
        color_column_values = [str(e) for e in df[color_column].unique()]
        palette = d3['Category10'][min(max(len(color_column_values), 3), 10)]
        kwargs['color'] = factor_cmap(color_column, palette=palette, factors=color_column_values)
        kwargs['legend'] = color_column

    use_jitter = True
    y = jitter(category_column, width=0.05, range=p.y_range, distribution='normal') if use_jitter else category_column

    p.circle(source=source, x=dof_column, y=y, alpha=0.5, **kwargs)

    p.legend.location = 'bottom_center'
    p.legend.orientation = "horizontal"
    p.legend.label_text_font_size = "6pt"
    p.legend.padding = 0
    p.legend.margin = 0

    if enable_hover:
        columns_to_hide = ['all_measurements', color_column, category_column, dof_column]
        hover = HoverTool()
        hover.tooltips = [(c, '@' + c) for c in df.columns if str(c) not in columns_to_hide]
        # hover.tooltips.append(('index', '$index'))
        p.add_tools(hover)
    return p, source
