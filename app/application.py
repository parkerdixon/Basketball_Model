# all the imports
import urllib.parse
import base64
import io
import pandas as pd
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash, send_file
from flask_paginate import Pagination, get_page_args
import networkx as nx
import matplotlib
from sklearn.metrics import pairwise

matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt

application = Flask(__name__) # create the application instance :)
application.config.from_object(__name__) # load config from this file , basketballapplication.py

indivStatCols = ['colmp',
 'colfg',
 'colfga',
 'colfgp',
 'col2p',
 'col2pa',
 'col2pp',
 'col3p',
 'col3pa',
 'col3pp',
 'colft',
 'colfta',
 'colftp',
 'coltrb',
 'colast',
 'colstl',
 'colblk',
 'coltov',
 'colfouls',
 'colpts',
 'colsos',
 'height',
 'weight',
 'teamfg']


data = pd.read_csv('draft-2018-projections.csv')
allCols = data.copy()
for col in data.columns:
    if ('percentile' in col) or ('player_x' != col and ('pred_' not in col)):
        data.drop(columns=[col], inplace=True)
data.set_index(['player_x'], inplace=True)
data.index.name=None

allSim = list(allCols.player1)+list(allCols.player2)+list(allCols.player3)+list(allCols.player4)+list(allCols.player5)
allPlayers = list(allCols.player_x)*5
simPlayers = dict(zip(range(len(allPlayers)),allSim))
allPlay = dict(zip(range(len(allPlayers)),allPlayers))
simSeries = pd.Series(simPlayers)
playerSeries = pd.Series(allPlay)
playerDf = pd.DataFrame(dict(playerSeries = playerSeries, simSeries = simSeries))
playerDf.columns = ['player','similar']

application.config.from_envvar('basketballapp_SETTINGS', silent=True)


@application.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the request."""
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()

@application.route('/')
def init_home():
    return render_template('layout.html')

def get_table(offset=0, per_page=100):
    return data.sort_values('pred_gamescore',ascending=False)[offset: offset + per_page]

@application.route('/all-players/')
def show_players():
    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
    total = len(data)
    table_rows = get_table(offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,css_framework='bootstrap4')
    return render_template('table-view.html',tables=[table_rows.to_html(classes='draft-data',table_id='all-players-table')],
    titles = ['na', 'Prospect Rankings'],pagination=pagination,page=page,per_page=per_page, cols = table_rows.columns)


@application.route('/sources')
def source_page():
    return render_template('sources.html')




@application.route('/player-plot', methods = ['POST'])
def plot_player_stats():
    playerName = request.form['playername']
    playerName = playerName.lower()
    playerList = list(allCols.player_x)
    playerListLower = [x.lower() for x in playerList]
    # if playerName not in playerListLower:
    #     abort(404)
    playerInd = playerListLower.index(playerName)
    playerName = playerList[playerInd]
    predDf = allCols[allCols.player_x==playerName]
    simPlays = []
    ranks = []
    for i in range(1,6):
        simPlays.extend(list(predDf['player' + str(i)]))
        ranks.append(float(predDf['score' + str(i) + '_rank']))
    simDf = pd.DataFrame({'Players':simPlays,'Similarity':ranks})
    simDf.set_index(['Players'], inplace=True)
    simDf.index.name=None
    predictionCols = []
    statVals = []
    statPerc = []
    for colName in predDf.columns:
        if 'pred_' in colName and 'percentile' not in colName and  'rookfgp' not in colName:
            predictionCols.append(colName)
            statVals.append(round(float(predDf[colName]),3))
            statPerc.append(float(predDf[colName+'percentile']))

    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(30, 15))
    fig.set_facecolor('white')
    ax.fontsize = 20
#     print(statPerc)
    ax.bar(range(len(predictionCols)), statPerc,color='#4863A0')
    predictionCols = ['3P%','2P%', 'Effective FG%', 'FT%','Rebounds','Assists','Steals','Blocks','Turnovers','Points','Gamescore']
    ax.set_xticklabels(predictionCols,fontsize=15,ha="center")
    ax.set_xticks(range(len(predictionCols)))
    ax.set_ylim((0,1.1))
    ax.set_ylabel('Historic Percentile',fontsize=15)
    ax.set_title(f'{playerName}: Rookie Year Predictions', color='black',fontsize=25)
    #     for i, v in enumerate(newDf['pred_' + curr_i]):
    #         ax.text(v, i, '  ' + str(round(v,3)), color='black',fontsize=20)

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals],fontsize=15,ha="right")

    rects = ax.patches
    # Make some labels.
    labels = list(statVals)

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, label,
                ha='center', va='bottom',fontsize=18)

    img = io.BytesIO()  # create the buffer
    fig.savefig(img, format='png')  # save figure to the buffer
    img.seek(0)  # rewind your buffer
    plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode()) # base64 encode & URL-escape

    thePlayer = playerName

    allCols['scores'] = pairwise.euclidean_distances(allCols[allCols.player_x == thePlayer][indivStatCols],allCols[indivStatCols])[0]
    similarPlayers = allCols.sort_values('scores').head(6)
    print(similarPlayers)
    toInclude = list(playerDf[playerDf.player.isin(similarPlayers.player_x)]['player'].unique())
    sizeScalar = len(toInclude)

    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(30,30))

    fig.set_facecolor('white')
    theDf = playerDf[playerDf.player.isin(toInclude)]
    src = 'player'
    targ = 'similar'
    g = nx.from_pandas_edgelist(theDf, source=src, target=targ)
    players = list(theDf[src].unique())
    simPlayers = list(theDf[targ].unique())

    # 2. Create a layout for our nodes
    layout = nx.spring_layout(g,iterations=50)

    # 3. Draw the parts we want
    nx.draw_networkx_edges(g, layout, width=3, edge_color='#566D7E',ax=ax,font_size=20)

    clubs = [node for node in g.nodes() if node in simPlayers]
    size = [g.degree(node)*200*sizeScalar for node in g.nodes() if node in simPlayers]
    nx.draw_networkx_nodes(g, layout, nodelist=clubs, node_size=size, node_color='#3CB371',ax=ax,font_size=20)

    people = [node for node in g.nodes() if node in players]
    nx.draw_networkx_nodes(g, layout, nodelist=people, node_size=100*sizeScalar, node_color='#fc8d62',ax=ax,font_size=20)
    nx.draw_networkx_nodes(g, layout, nodelist=[thePlayer], node_size=8000, node_color='#e63920',ax=ax,font_size=20)

    club_dict = dict(zip(players+simPlayers,players+simPlayers))
    nx.draw_networkx_labels(g, layout, labels=club_dict,ax=ax,font_size=20)

    # 4. Turn off the axis because I know you don't want it
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(f'Similarity Web for {thePlayer}',fontsize=30)

    img2 = io.BytesIO()  # create the buffer
    fig.savefig(img2, format='png')  # save figure to the buffer
    img2.seek(0)  # rewind your buffer
    sim_data = urllib.parse.quote(base64.b64encode(img2.read()).decode()) # base64 encode & URL-escape

    return render_template('player-plot.html', plot_url=plot_data,sim_url=sim_data,tables=[simDf.to_html(classes='draft-data')],titles='Similar Players')

@application.route('/search', methods=['GET','POST'])
def search_page():
    players = list(allCols.player_x)
    players.sort()
    return render_template('search.html', players=players)


if __name__ == '__main__':
    application.run()
