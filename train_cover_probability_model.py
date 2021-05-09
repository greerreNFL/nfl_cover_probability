## basic imports ##
import pandas as pd
import numpy

## ml imports ##
## A lot of the below follows this guide youtube.com/watch?v=GrJP9FLV3FE&t=407s ##
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss

## output ##
output_folder = 'A LOCAL FILEPATH FOR SAVING THE MODEL AND EVALUATION METRICS'

## load pbp data ##
pbp_filepath = 'A LOCAL FILEPATH WHERE YOUR PBP DATA LIVES. YOU CAN ALSO READ THIS DIRECTLY FROM GITHUB'
pbp_df = pd.read_csv(pbp_filepath, low_memory=False, index_col=0)




## create some new variables for the model ##
## most features taken directly from nflfastR ##

## SPREAD_LINE_DIFFERENTIAL ##
## instead of a point differential, use a spread line differential ##
## ie how close is the team to covering ##
pbp_df['spread_line_differential'] = numpy.where(
    pbp_df['posteam_type'] == 'home',
    -1 * pbp_df['spread_line'] + pbp_df['score_differential'],
    numpy.where(
        pbp_df['posteam_type'] == 'away',
        pbp_df['spread_line'] + pbp_df['score_differential'],
        numpy.nan
    )
)

## elapsed share, spread_time, and Diff_Time_Ratio are all custom features from nflfastR's model ##
## https://raw.githubusercontent.com/mrcaseb/nflfastR/master/R/helper_add_ep_wp.R ##

## elapsed share ##
pbp_df['elapsed_share'] = (
    (3600 - pbp_df['game_seconds_remaining']) / 
    3600
)

pbp_df['posteam_spread'] = numpy.where(
    pbp_df['posteam_type'] == 'home',
    pbp_df['spread_line'],
    -1 * pbp_df['spread_line']
)

## spread_time ##
pbp_df['spread_time'] = pbp_df['posteam_spread'] * numpy.exp(-4 * pbp_df['elapsed_share'])

## Diff_Time_Ratio ##
pbp_df['diff_time_ratio'] = pbp_df['score_differential'] / numpy.exp(-4 * pbp_df['elapsed_share'])


## RECEIVE_2H_KO ##
## determine who received the first kickoff ##
kickoff_df = pbp_df[pbp_df['play_type'] == 'kickoff'].groupby(
    ['game_id']
)[['game_id','posteam_type']].head(1)

## add back to df ##
pbp_df = pd.merge(
    pbp_df,
    kickoff_df.rename(columns={
        'posteam_type' : 'received_first_ko'
    }),
    on=['game_id'],
    how='left'
)

## create receive 2nd half ko variable ##
pbp_df['receive_2h_ko'] = numpy.where(
    (pbp_df['game_half'] == 'Half1') &
    (pbp_df['posteam_type'] != pbp_df['received_first_ko']),
    1,
    0
)


## IS_PAT ##
## denote if a play is a pat ##
pbp_df['is_pat'] = numpy.where(
    pbp_df['play_type'] == 'extra_point',
    1,
    0
)


## POSTEAM_IS_HOME ##
## turn posteam_type into a boolean ##
pbp_df['posteam_is_home'] = numpy.where(
    pbp_df['posteam_type'] == 'home',
    1,
    numpy.where(
        pbp_df['posteam_type'] == 'away',
        0,
        numpy.nan
    )
)


## COVER_RESULT ##
pbp_df['cover_result'] = numpy.where(
    pbp_df['posteam_type'] == 'home',
    numpy.where(
        -1 * pbp_df['spread_line'] + pbp_df['result'] > 0,
        1,
        0
    ),
    numpy.where(
        pbp_df['posteam_type'] == 'away',
        numpy.where(
            pbp_df['spread_line'] + -1 * pbp_df['result'] > 0,
            1,
            0
        ),
        numpy.nan
    )
)


## filter down to just the columns we need for the model ##
## preserving full pbp_df in case we want to use it later ##
## narrator: They did not, in fact, end up using it later ##
model_df = pbp_df[[
    ## only needed for train/test split ##
    'game_id',
    'season',
    ## dependent var ##
    'cover_result',
    ## independent vars from WP model ##
    'spread_time',
    'score_differential',
    'diff_time_ratio',
    'posteam_is_home',
    'half_seconds_remaining',
    'game_seconds_remaining',
    'down',
    'ydstogo',
    'yardline_100',
    'posteam_timeouts_remaining',
    'defteam_timeouts_remaining',
    'receive_2h_ko',
    ## new features for CP model ##
    'is_pat',
    'spread_line_differential',
]].copy()

## remove NAs ##
model_df = model_df.dropna()

## MODEL ##
## Split dependent and independent data frames ##
## since data is at the play level and we want to predict something that occurs
## at the game level, we can't just take a random sample of plays ##
## instead, we will take a random sample of games and apply test/train sets that way ##
## this ensures no game has plays in both the test and train ##

## we'll also hold out the last two seasons (2019 and 2020) for validation ##
model_construction_df = model_df[model_df['season'] < 2019].copy()
model_validation_df = model_df[model_df['season'] >= 2019].copy()


## get df of unique games ##
set_key_df = model_construction_df.groupby(['game_id'])['game_id'].head(1).reset_index()[['game_id']].copy()

## assign to test / train randomly ##
set_key_df['rand_float'] = numpy.random.uniform(
    low=0,
    high=1,
    size=len(set_key_df)
)

## assign set ##
set_key_df['is_training_set'] = numpy.where(
    set_key_df['rand_float'] > .33,
    1,
    0
)

## match back to model df ##
model_construction_df = pd.merge(
    model_construction_df,
    set_key_df[['game_id', 'is_training_set']],
    on=['game_id'],
    how='left'
)

## create training and test sets ##
training_df = model_construction_df[model_construction_df['is_training_set'] == 1].copy()
test_df = model_construction_df[model_construction_df['is_training_set'] == 0].copy()

## create x and y versions ##
X_train = training_df.drop(columns=['game_id', 'season', 'is_training_set', 'cover_result']).copy()
X_test = test_df.drop(columns=['game_id', 'season', 'is_training_set', 'cover_result']).copy()

y_train = training_df['cover_result'].copy()
y_test = test_df['cover_result'].copy()


## create first model to make sure evrything works ##
clf_xgb = xgb.XGBClassifier(objective='binary:logistic')
clf_xgb.fit(
    X_train,
    y_train,
    verbose=True,
    early_stopping_rounds = 10,
    eval_metric='aucpr',
    eval_set=[(X_test, y_test)]
)


## Hyperparameter Optimization ##
## do some hyper parameter optimization ##
## Round 1 ##
param_grid = {
    'max_depth' : [3, 4, 5],
    'learning_rate' : [0.1, 0.05, 0.01],
    'gamma' : [0, 0.25, .5],
    'reg_lambda' : [10, 12, 15],
    'n_estimators' : [100, 500, 1000],
}

## Round 1 Results ##
## {'gamma': 0.25, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1000, 'reg_lambda': 10}

## Round 2 ##
param_grid = {
    'max_depth' : [5, 6, 7],
    'learning_rate' : [0.005, 0.01, 0.025],
    'gamma' : [.25],
    'reg_lambda' : [6, 8, 10],
    'n_estimators' : [1000, 1250, 1500],
}

## {'gamma': 0.25, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1000, 'reg_lambda': 6}


## Round 3 ##
param_grid = {
    'max_depth' : [5],
    'learning_rate' : [0.01],
    'gamma' : [.25],
    'reg_lambda' : [2, 4, 6],
    'n_estimators' : [1000, 1125],
}

## {'gamma': 0.25, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1000, 'reg_lambda': 6}


## set up grid search ##
optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(
        objective='binary:logistic',
        subsample=0.9,
        colsample_bytree=0.75
    ),
    param_grid=param_grid,
    scoring='roc_auc',
    verbose=0,
    cv=3
)

## fit ##
optimal_params.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)]
)


## rerun w/ tuned params ##
clf_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    gamma=0.25,
    max_depth=5,
    reg_lambda=6,
    learning_rate=0.01,
    n_estimators=1000
)

clf_xgb.fit(
    X_train,
    y_train,
    verbose=True,
    early_stopping_rounds = 10,
    eval_metric='aucpr',
    eval_set=[(X_test, y_test)]
)


## save model for future use ##
clf_xgb.save_model(
    '{0}/cp.model'.format(output_folder)
)


## a function for saving csvs of model performance locally. Not necessary to run ##
def score_models(model_arrays):
    bin_dfs = []
    confusion_dfs = []
    metric_dfs = []
    for i in model_arrays:
        df = i[0].copy()
        output_name = i[1]
        ## create predictions ##
        df['cover_prob'] = clf_xgb.predict_proba(df.drop(columns=['cover_result']))[:,1]
        ## bins ##
        bins = numpy.linspace(0, 1, 100)
        binned_df = df.groupby(
            numpy.digitize(df['cover_prob'], bins)
        ).agg(
            cover_average = ('cover_result', 'mean'),
            observations = ('cover_result', 'count'),
        ).reset_index().rename(columns={
            'index' : 'cover_prob'
        })
        binned_df['set_type'] = output_name
        ## confusion ##
        df['true_pos'] = numpy.where(
            (df['cover_prob'] > .5) &
            (df['cover_result'] == 1),
            1,
            0
        )
        df['false_pos'] = numpy.where(
            (df['cover_prob'] > .5) &
            (df['cover_result'] == 0),
            1,
            0
        )
        df['true_neg'] = numpy.where(
            (df['cover_prob'] < .5) &
            (df['cover_result'] == 0),
            1,
            0
        )
        df['false_neg'] = numpy.where(
            (df['cover_prob'] < .5) &
            (df['cover_result'] == 1),
            1,
            0
        )
        confusion_df = pd.DataFrame([{
            'set_type:' : output_name,
            'true_positive' : df['true_pos'].sum(),
            'false_positive' : df['false_pos'].sum(),
            'true_negative' : df['true_neg'].sum(),
            'false_negative' : df['false_neg'].sum(),
        }])
        ## log loss ##
        log_loss_score = log_loss(
            df['cover_result'],
            df['cover_prob']
        )
        auc = roc_auc_score(
            df['cover_result'],
            df['cover_prob']
        )
        metric_df = pd.DataFrame([{
            'set_type:' : output_name,
            'log_loss' : log_loss_score,
            'roc_auc' : auc,
        }])
        bin_dfs.append(binned_df)
        confusion_dfs.append(confusion_df)
        metric_dfs.append(metric_df)
    bin_output = pd.concat(bin_dfs)
    confusion_output = pd.concat(confusion_dfs)
    metrics_output = pd.concat(metric_dfs)
    ## output ##
    bin_output.to_csv(
        '{0}/binned_results.csv'.format(
            output_folder
        )
    )
    confusion_output.to_csv(
        '{0}/confusion_results.csv'.format(
            output_folder
        )
    )
    metrics_output.to_csv(
        '{0}/metric_results.csv'.format(
            output_folder
        )
    )


test_arrays = [
    [
        training_df.drop(columns=['game_id', 'season', 'is_training_set']).copy(),
        'training'
    ],
    [
        test_df.drop(columns=['game_id', 'season', 'is_training_set']).copy(),
        'test'
    ],
    [
        model_validation_df.drop(columns=['game_id', 'season']).copy(),
        'validate'
    ],
]


score_models(test_arrays)
