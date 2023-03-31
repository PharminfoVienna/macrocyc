import base64
import numpy as np
import random
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, dcc, html, no_update, Output, Input, dash
import plotly.graph_objects as go
from dash.dcc import Markdown
import math

#info = {"SLCO":"SLCO is "}

navbar = dbc.Navbar(
    [
        dbc.Col(
            html.Img(src="data:image/svg+xml;base64," + base64.b64encode(open('logo.svg', 'rb').read()).decode('utf-8'), height="70px"),
            width={"size": "3" }
        ),
        dbc.Col(
            html.H2("The Macrocycle Inhibitor Landscape of SLC-Transporters", className="text-center text-white"),
            width={"size": "10", "offset": 0}
        )
    ],
    color="#0063a6ff",
    dark=True
)

def dropdown(df):
    return dcc.Dropdown(id='dropdown',
                        options=[
                            {'label': i, 'value': i} for i in df['family'].unique()
                        ] + [{'label': 'pIC50', 'value': 'pchembl_value'}],
                        placeholder="Select a SLC family or switch to the IC50 mode")


def colors_explained(data):
    data = data[['family','color']].drop_duplicates()
    #Create an invisible table dash
    table = dbc.Table.from_dataframe(data,id="slc-legend", striped=False, borderless=True, hover=False, responsive=True)
    del (table.children[0])
    for tr in table.children[0].children:
        first, second = tr.children
        #Squared element with color and family name
        first.style = {'width':'10%', 'height':'100%'}
        tmp = str(first.children)
        first.children = html.Div(tmp,id=f"info-{tmp}")
        second.children = html.Div("|",style={"overflow":"hidden","border-radius": "10px","width":"15%", "height":"100%", "background-color": second.children,"color": second.children}),

        table.children.append(dbc.Tooltip(
            f"Info about the {tmp} family",
            target=f"info-{tmp}",
        ))



    #tags = []
    #for (_,color) in data.iterrows():
    #    tags.append(dbc.Label(color['family'], style={'background-color': color['color']}))
    return table

def color_space(n):
    hue_list = range(0,360,360/n)
    return ['hsl(' + str(l) + ',100%,50%)' for l in hue_list]
rand_color = lambda : "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])




def figure(data):
    return go.Figure(data=[
        go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers",
            marker=dict(
                colorscale='viridis',
                size=4,
                colorbar=None,#{"title": "Loss"},
                line={"color": "#444"},
                reversescale=True,
                sizeref=45,
                sizemode="diameter",
                opacity=0.8,
            )
        )
])




if __name__ == '__main__':
        app = dash.Dash(
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
        data = pd.read_csv('new.csv')
        images = pd.read_csv('images.csv', index_col=0)
        #Drop the rows for which standard_type is not IC50
        subset = data[data['standard_type'] == 'IC50']
        atpair_list = [np.nan] * len(data)
        for i, row in subset.iterrows():
            atpair_list[i] = (row['family'], row['pchembl_value'])
        data['atpair'] = atpair_list
        print(data)
        #data['atpair'] = list(zip(subset['family'], subset['pchembl_value']))
        grouped = data.groupby('canonical_smiles').agg({'atpair': lambda x: x.tolist(), 'x': 'mean', 'y': 'mean','pchembl_value':'median','color':'first'}).reset_index()
        fig = figure(grouped[['x', 'y']])
        fig.update_traces(hoverinfo="none", hovertemplate=None)
        fig.update_layout(
            xaxis=dict(title='X - Coordinate'),
            yaxis=dict(title='Y - Coordinate'),
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=10,
                pad=4
            )
        )

        app.layout = html.Div(className='document', children=[
            dbc.Row(navbar),
            dbc.Row([
                dbc.Col(Markdown("""
                #### Abstract
                This is a visualisation for the manuscript **"The macrocycle
                inhibitor landscape of SLC-transporter"**. The visualisation is
                based on MolCompass: a parametric t-SNE approach for the
                visualisation of chemical space. The general idea is to project
                structurally similar molecules into the same region of the 2D map.
                #### Cite us
                Nejra Granulo, Sergey Sosnin, Gerhard F. Ecker, *The macrocycle
                inhibitor landscape of SLC-transporter*, sumbitted to Molecular
                Informatics
                #### Questions?
                Please contact us at: mailto:gerhard.f.ecker@univie.ac.at
                """,style={'font-size': '18px','margin-left':'20px','text-align':'justify'}), width={"size": 3},align="center"),

                dbc.Col([
                    dbc.Row(dropdown(data), justify="center",style={'margin-top':'20px','margin-left':'30px'}),
                    dbc.Row([
                            dcc.Graph(id='graph-basic-2', figure=fig, clear_on_unhover=True,
                                      style={'height': '80vh'}),
                            dcc.Tooltip(id="graph-tooltip")
                        ],style={'margin-top':'20px'},justify="center"),
                ],width={"size": 6},align="center"),

                dbc.Col([
                    colors_explained(data),
                ], align="center")

            ])
        ])

        @app.callback(Output('graph-basic-2', 'figure'),
                      Output('slc-legend', 'style'),
                      [Input('dropdown', 'value')])
        def update_figure(dropdown_value):
            slc_legend_visible = {'visibility': 'visible'} if not dropdown_value else {'visibility': 'hidden'}
            colorbar = None
            if dropdown_value != 'pchembl_value' and dropdown_value is not None:
                selected_index = grouped['canonical_smiles'].isin(data[data['family'] == dropdown_value]['canonical_smiles'])
                color = selected_index.apply(lambda x: 'red' if x else 'grey')
                opacity = selected_index.apply(lambda x: 1 if x else 0.4)
                #fig.update_layout(coloraxis_showscale=False)
                #red_points = selected_index
            elif dropdown_value == 'pchembl_value':
                #selected = data
                colorbar = {'title':"pIC50"}
                #fig.update_layout(coloraxis_showscale=True)
                color=[
                    'gray' if math.isnan(x) else x
                    for x in data['pchembl_value']
                ]
                #color=grouped['pchembl_value']
                opacity = 0.8
                pass
            elif dropdown_value is None:
               # fig.update_layout(coloraxis_showscale=False)
                color = grouped['color']
                opacity = 0.8
                pass
            if colorbar is None:
                fig.update_traces(x=grouped["x"], y=grouped["y"], mode="markers", marker=dict(
                    colorscale='viridis',
                    color=color, #data['pchembl_value'] if dropdown_value == 'pchembl_value' else data['color'] if not dropdown_value else
                    #data['family'].apply(lambda x: 'red' if x == dropdown_value else 'grey'),
                    size=5,
                    line={"color": "#444"},
                    reversescale=True,
                    sizeref=45,
                    sizemode="diameter",
                    opacity=opacity,
                ))
                fig.update_layout(showlegend=False)
            else:
                fig.update_traces(x=grouped["x"], y=grouped["y"], mode="markers", marker=dict(
                    colorscale='viridis',
                    color=color, #data['pchembl_value'] if dropdown_value == 'pchembl_value' else data['color'] if not dropdown_value else
                    #data['family'].apply(lambda x: 'red' if x == dropdown_value else 'grey'),
                    size=5,
                    colorbar=colorbar,
                    line={"color": "#444"},
                    reversescale=True,
                    sizeref=45,
                    sizemode="diameter",
                    opacity=opacity,
                ))
            return fig,slc_legend_visible


        @app.callback(
            Output("graph-tooltip", "show"),
            Output("graph-tooltip", "bbox"),
            Output("graph-tooltip", "children"),
            Input("graph-basic-2", "hoverData"),
        )
        def display_hover(hoverData):
            if hoverData is None:
                return False, no_update, no_update

            # demo only shows the first point, but other points may also be available

            # Find all points with the same x and y coordinates in data
            # data_subset = data[(data['x'] == hoverData['points'][0]['x']) & (data['y'] == hoverData['points'][0]['y'])]
            #data_subset = data[(data['x'] == hoverData['points'][0]['x']) & (data['y'] == hoverData['points'][0]['y'])]
            #print(list(data_subset.iterrows()))
            # print("----------------------------------------------------")
            pt = hoverData["points"][0]
            data_subset = grouped[(grouped['x'] == pt['x']) & (grouped['y'] == pt['y'])]
            # point = data.loc[(pt['pointNumber']), ['canonical_smiles', 'pchembl_value','family']]

            bbox = pt["bbox"]

            def make_family_record(family, records,median,range):
                v1 = html.Div(
                    [
                        html.P(f'Family: {family}'),
                        html.P(f'Median value: {round(median,3)}'),
                        html.P(f'No of Records: {records}'),
                        html.P(f'Range: {range}'),
                        html.Hr(style={"margin": "10px 0px"})
                    ]
                )
                v2 = html.Div(
                    [
                        html.P(f'Family: {family}'),
                        html.P(f'Value: {round(median,3)}'),
                        html.P(f'No of Records: {records}'),
                        html.Hr(style={"margin": "10px 0px"})
                    ]
                )
                return v1 if records > 1 else v2

            def make_molecular_card(subset):
                datapoint = pd.DataFrame(subset.atpair,columns=['family','value'])
                datapoint = datapoint.groupby('family').agg(Median=('value','median'),Records=('value','count'),Range=('value',lambda x: f'{x.min()} - {x.max()}')).reset_index()#{'value': {'median': 'median', 'range': lambda x: (x.min(), x.max())}})
                return html.Div([
                    html.Img(src=images.loc[subset.canonical_smiles].values[0], style={'width': '100%', 'height': '100%'}),
                    html.Div([make_family_record(p['family'], p['Records'],p['Median'],p['Range']) for p in datapoint.to_dict(orient="records")]),
                ])

            children = html.Div([
                make_molecular_card(point[1]) for point in data_subset[['canonical_smiles','atpair']].iterrows()
            ], style={'font-size': '18px', 'display': 'inline','width':'100px','max-width':'50vh'})
            return True, bbox, children

        app.run_server(debug=False,host='0.0.0.0',port=8050)

