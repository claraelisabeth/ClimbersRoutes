# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt

# colors
even_lighter_beige = 'rgb(247, 242, 233)'
light_beige = 'rgb(250, 240, 220)'
beige = 'rgb(213, 184, 149)'
brown = 'rgb(86, 56, 46)'
light_brown = 'rgb(113, 80, 58)'
light_green = 'rgb(168, 203, 160)'
green = 'rgb(78, 110, 77)'

# colors and order for grades
grade_order = [
    "3", "4a", "4a+", "4b", "4b+", "4c", "4c+", "5a", "5a+", "5b", "5b+", "5c", "5c+",
    "6a", "6a+", "6b", "6b+", "6c", "6c+", "7a", "7a+", "7b", "7b+", "7c", "7c+",
    "8a", "8a+", "8b", "8b+", "8c", "8c+", "9a", "9a+", "9b", "9b+", "9c", "9c+", 
    'a3', 'a4', 'd', 'ed1', 'ed2', 'ed3', 'f-', 'td', 'td+'
    ]

grade_groups = {
     "3&4": ["3", "4a", "4a+", "4b", "4b+", "4c", "4c+"],
     "5": ["5a", "5b", "5b+", "5c", "5c+"],
     "6": ["6a", "6a+", "6b", "6b+", "6c", "6c+"],
     "7": ["7a", "7a+", "7b", "7b+", "7c", "7c+"],
     "8": ["8a", "8a+", "8b", "8b+", "8c", "8c+"],
     "9": ["9a", "9a+", "9b", "9b+", "9c"],
     "rest": ['a3', 'a4', 'd', 'ed1', 'ed2', 'ed3', 'f-', 'td', 'td+']
     }

group_colors = {
    "3&4": green,
    "5": 'rgb(121,149,103)',
    "6": 'rgb(167,181,158)',
    "7": 'rgb(249,221,216)',
    "8": 'rgb(248,208,200)',
    "9": 'rgb(243,186,186)',
    "rest": brown
    }
grade_color_map = {grade: group_colors[group] for group, grades in grade_groups.items() for grade in grades}



# Data

df = pd.read_excel('data/climbing_history_all_cleanish.xlsx')
style = []
weight = []

for style_weight in df['Style']:
    if '|' in style_weight:
        style_weight_split = style_weight.split('|')
        style.append(style_weight_split[0].strip())
        weight.append(style_weight_split[1].strip())
    else:
        style.append(style_weight_split[0].strip())
        weight.append(np.nan)

df['climbing_style'] = style
df['weight'] = weight

df.rename(columns={
    'Climber': 'climber', 
    'Official grade': 'official_grade',
    'Route' : 'route',
    'Suggested Grade' : 'suggested_grade',
    'Ascent Date' : 'ascent_date'},
    inplace=True)

df = df.drop(['Style'], axis=1)
df['date'] = pd.to_datetime(df['parsed_dates'], format='mixed', errors='coerce')

grade_mapping = {
                "vd" : "4a",
                "s" : "4b",
                "hs" : "4c",
                "vs" : "5a",
                "hvs" : "5b",
                "e1" : "5c",
                "e2" : "6a+",
                "e3" : "6b+",
                "e4" : "6c+",
                "e5" : "7a+",
                "e6" : "7b+",
                "e7" : "7c+",
                "e8" : "8a+",
                "e9" : "8b",
                "e10" : "8b+",
                "e11" : "8c+",
                "e12" : "9a",
                "5+" : "5c"
                }

df['official_grade_consistent'] = df['official_grade'].str.replace(" (approx)", "", regex=False)
df['official_grade_consistent'] = df['official_grade_consistent'].replace(grade_mapping)

# wrongly classified routes
df.loc[df['route'] == 'Central Pillar Of Freney', 'climbing_style'] = 'Mountaineering'
df.loc[df['route'] == 'Atlantic Ocean Wall', 'climbing_style'] = 'Aid'
df.loc[df['route'] == 'Lafaille Route', 'climbing_style'] = 'Aid'
df.loc[df['route'] == 'Flight Of The Albatross', 'climbing_style'] = 'Aid'
df.loc[df['route'] == 'Zenyatta Mondatta', 'climbing_style'] = 'Aid'




# Creating all M_i

M_dict = {}

for route_type, group in df.groupby('climbing_style'):
    climbers = group['climber'].unique()
    routes = group['route'].unique()
    
    climber_to_index = {climber: c for c, climber in enumerate(climbers)}
    route_to_index = {route: r for r, route in enumerate(routes)}
    
    M_temp = np.zeros((len(climbers), len(routes)), dtype=int)
    
    climber_indices = group['climber'].map(climber_to_index)
    route_indices = group['route'].map(route_to_index)
    
    M_temp[climber_indices, route_indices] = 1
    M_dict[route_type] = M_temp


# Sorting the M_i
sorted_byRow_M_dict = {}
sorted_byColumn_M_dict = {}

for style in df['climbing_style'].unique():
    byRow_order = np.argsort(np.count_nonzero(M_dict[style], axis=1))
    sorted_byRow_M_temp = M_dict[style][byRow_order]
    sorted_byRow_M_dict[style] = sorted_byRow_M_temp

    byColumn_order = np.argsort(np.count_nonzero(M_dict[style], axis=0))
    sorted_byColumn_M_temp = M_dict[style][:, byColumn_order]
    sorted_byColumn_M_dict[style] = sorted_byColumn_M_temp





# Helper Functions

def climbing_styles_distribution(style_list = df['climbing_style'].unique()):
    route_counts = df[df['climbing_style'].isin(style_list)].groupby(['climbing_style', 'official_grade_consistent']).size().unstack(fill_value=0)
    #route_counts['total'] = route_counts.sum(axis=1)

    route_counts_long = route_counts.reset_index().melt(
        id_vars='climbing_style', 
        var_name='official_grade_consistent', 
        value_name='route_count'
    )

    route_counts_long['official_grade_consistent'] = pd.Categorical(
        route_counts_long['official_grade_consistent'], 
        categories=grade_order, 
        ordered=True
    )
    route_counts_long = route_counts_long.sort_values('official_grade_consistent')

    fig = px.bar(
        route_counts_long,
        x='climbing_style',
        y='route_count',
        color='official_grade_consistent',
        color_continuous_scale='tealrose',
        title='Route Distribution',
        labels={'climbing_style': 'Climbing Type', 'route_count': 'Number of Routes'},
        color_discrete_map=grade_color_map
    )
    fig.update_layout(
        xaxis_title="Climbing Type",
        yaxis_title="Number of Routes",
        legend_title="Grade",
        xaxis=dict(tickangle=90), 
        title=dict(font=dict(size=16)),
        legend=dict(font=dict(size=10)),
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        title_font_color='rgb(78, 110, 77)',
    )
    return fig



def creating_bipartite_network(M_dict, route_type, climber = 'All', route = 'All'):
    M = M_dict[route_type]

    if climber == 'All':
        climbers = list(df[df['climbing_style'] == route_type]['climber'].unique())
    else:
        climbers = [climber]

    if route == 'All':
        routes = list(df[df['climbing_style'] == route_type]['route'].unique())
    else:
        routes = [route]
    
    B = nx.Graph()

    # Add nodes for climbers (layer 0) and routes (layer 1)
    B.add_nodes_from(climbers, bipartite=0)  
    B.add_nodes_from(routes, bipartite=1) 

    for c, climber in enumerate(climbers):
        for r, route in enumerate(routes):
            if M[c, r] == 1:
                B.add_edge(climber, route)
    return B



def visualize_bipartite_px(B, climbing_type):
    climbers = {n for n, d in B.nodes(data=True) if d['bipartite'] == 0}
    routes = set(B) - climbers

    climber_positions = {climber: (0, i) for i, climber in enumerate(sorted(climbers))}
    route_positions = {route: (1, i) for i, route in enumerate(sorted(routes))}
    positions = {**climber_positions, **route_positions}

    edge_x = []
    edge_y = []
    for edge in B.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x += [x0, x1, None]  
        edge_y += [y0, y1, None]

    node_x = []
    node_y = []
    node_color = []
    node_hover = []

    for node, (x, y) in positions.items():
        node_x.append(x)
        node_y.append(y)
        node_color.append('lightblue' if node in climbers else 'lightgreen')
        node_hover.append(node)  # Show node name on hover

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(color='gray', width=1),
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(size=10, 
                    color=node_color,
                    opacity=0.8,
                    line=dict(width=2,
                              color='DarkSlateGrey')),
        text=node_hover,  # Show node names
        hoverinfo='text'
    ))
    fig.update_layout(
        title=f"Bipartite Network for {climbing_type}",
        xaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
        showlegend=True,
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        title_font_color='rgb(78, 110, 77)',
    )
    return fig

def top_climbers():
    route_counts = df.groupby(['climber', 'climbing_style']).size().unstack(fill_value=0)
    route_counts['total'] = route_counts.sum(axis=1)

    top_climbers = route_counts.nlargest(15, 'total')  
    route_counts = top_climbers.drop(columns='total')  

    route_counts_long = route_counts.reset_index().melt(
        id_vars='climber', 
        var_name='climbing_style', 
        value_name='route_count'
    )
    fig = px.bar(
        route_counts_long,
        x='climber',
        y='route_count',
        color='climbing_style',
        title='Top (climbed the most) 15 Climbers',
        labels={'climber': 'Climber', 'route_count': 'Number of Routes'},
    )
    fig.update_layout(
        xaxis_title="Climber",
        yaxis_title="Number of Routes",
        legend_title="Climbing Type",
        xaxis=dict(tickangle=90), 
        title=dict(font=dict(size=16)),
        legend=dict(font=dict(size=10))
    )
    return fig


def spy_plotly(matrix, title=None):
    rows, cols = np.nonzero(matrix)
    fig = go.Figure(data=go.Scatter(
        x=cols, 
        y=rows,
        mode='markers',
        marker=dict(size=5, color='rgb(78, 110, 77)') 
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title='Routes'),
        yaxis=dict(title='Climbers', autorange='reversed'),
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        title_font_color='rgb(78, 110, 77)',
    )
    return fig






# Initialize the app - incorporate css
app = Dash(__name__, external_stylesheets=['../static/css/styles.css'])

# App layout
app.layout = html.Div([
    html.Div([
        # Title and Logo
        html.Div("Climbers and Routes - EFC Analysis", id='title'),
        html.Div("Clara Pichler", id='user-name'),
        html.Img(src='./static/Raccoon.png',
                 id='racoon-logo', height='60px', width='60px'),
    ], id='header', style={'display': 'flex', 'align-items': 'center'}),

    html.Div(className='overview-box', children=[
        html.Div(id='overview-text', 
                 children="""Description I might add later on""")
    ]),

    html.Div(className='main-box', children=[
        html.Div(dcc.Graph(figure={}, id='main-graph'))
    ]),

    html.Div(id='change-parameters', className='parameter-box', children=[
        html.Div(children="Parameter Control", 
                 style={'color': 'rgb(78, 110, 77)', 'font-size': '25px', 'padding':'20px'}),
        html.Div(className='dropdown', children=[
            dcc.Dropdown(
                id='style_dropdown',
                options=[{'label': str(style), 'value': style} for style in df['climbing_style'].unique()],
                value='Solo',
                placeholder="Select Climbing Style"
            )
        ]),
    ]),

    html.Div(dcc.Graph(figure={}, id='spy-sorted-byRow')),

    html.Div(dcc.Graph(figure={}, id='spy-sorted-byColumn')),

    html.Div(dcc.Graph(figure={}, id='spy-not-sorted')),

    html.Div(dcc.Graph(figure=climbing_styles_distribution(['Boulder', 'Lead']), id='barplot_styles'))
])


# Controls to build the interaction
# app.callback

@callback(
    [Output('main-graph', 'figure'),
     Output('spy-sorted-byRow', 'figure'),
     Output('spy-sorted-byColumn', 'figure'),
     Output('spy-not-sorted', 'figure')
    ],
    [Input('style_dropdown', 'value')
    ]
)
def update_graph(selected_style):

    try:
        network = creating_bipartite_network(M_dict, selected_style)
        return [visualize_bipartite_px(network, selected_style), 
                spy_plotly(sorted_byRow_M_dict[selected_style], 'Spy Plot of sorted (by row) M - ' + selected_style ), 
                spy_plotly(sorted_byColumn_M_dict[selected_style], 'Spy Plot of sorted (by column) M - ' + selected_style ),
                spy_plotly(M_dict[selected_style], 'Spy Plot of M - ' + selected_style)]
    
    except Exception as e:
        print(f"Error in callback: {e}")
        return [go.Figure()]

if __name__ == '__main__':
    app.run(debug=True)