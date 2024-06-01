import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from io import StringIO
import logging
import base64
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report, auc
import plotly.graph_objs as go
import joblib
from plotly.subplots import make_subplots



# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app with suppress_callback_exceptions=True
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Delirium Dashboard"),  # Header
    html.H3("Upload your preprocessed (discrete time) data!"),  # Title
    
    # Upload section
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px 0'
        },
        # Allow only one file to be uploaded
        multiple=False
    ),
    
    # Store for uploaded data
    dcc.Store(id='uploaded-data'),
      # Store for training completion flag
    dcc.Store(id='training-completed', data=False),
    
    # Output for displaying uploaded data
    html.Div(id='output-data-upload'),

    # Placeholder for ROC Curve graph
    html.Div(id='roc-curve-container', style={'display': 'none'}, children=[
        dcc.Graph(id='roc-curve')
    ])
])


# Callback to store uploaded data
@app.callback(Output('uploaded-data', 'data'),
              [Input('upload-data', 'contents')])
def store_uploaded_data(contents):
    if contents is not None:
        logger.info("Data uploaded.")
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        decoded_str = decoded.decode('utf-8')
        return decoded_str
    else:
        logger.warning("No data uploaded.")
        return None


# Callback to parse uploaded data and generate UI elements
@app.callback(Output('output-data-upload', 'children'),
              [Input('uploaded-data', 'data')])
def parse_uploaded_data(data):
    if data is not None:
        logger.info("Parsing uploaded data...")
        decoded_df = pd.read_csv(StringIO(data))
        
        # Generate checkbox options for features (X variables)
        feature_options = [{'label': col, 'value': col} for col in decoded_df.columns]
        
        # Generate dropdown options for target variable (y variable)
        target_options = [{'label': col, 'value': col} for col in decoded_df.columns]

        logger.info("Data parsing completed.")
        
        # Determine if scroll is needed
        scroll_needed = len(feature_options) > 15
        
        return html.Div([
            # Display feature checkbox list with scroll if needed
            html.Div(
                dcc.Checklist(
                    id='features-checkbox',
                    options=feature_options,
                    labelStyle={'display': 'block'}
                ),
                style={'maxHeight': '300px', 'overflowY': 'scroll'} if scroll_needed else {}
            ),
            html.Br(),
            # Display target dropdown
            dcc.Dropdown(
                id='target-dropdown',
                options=target_options,
                placeholder="Select target variable"
            ),
            html.Br(),

            dcc.Dropdown(
                id = 'id-dropdown',
                options = target_options,
                placeholder= 'Select patient ID column'
            ),

            html.Br(),
            # Train model button
            html.Button('Train model', id='train-model-button', n_clicks=0),

            html.H3('Model Summary'),

            # Hidden div to store the ROC curve graph
            html.Div(id='roc-curve-container', style={'display': 'none'}, children=[
                dcc.Graph(id='roc-curve')
            ])
        ])
    else:
        return None


# Callback to log selected checkboxes, target variable, and DataFrame head upon clicking the "Train model" button
@app.callback([Output('training-completed', 'data'), 
               Output('roc-curve', 'figure')],
              [Input('train-model-button', 'n_clicks'),
               Input('uploaded-data', 'data')],
              [State('features-checkbox', 'value'),
               State('target-dropdown', 'value'),
               State('id-dropdown', 'value')])
def train_model(n_clicks, data, selected_features, selected_target, selected_id):
    if n_clicks and data:
        logger.info("Training model...")
        logger.info("Selected features: %s", selected_features)
        logger.info("Selected target variable: %s", selected_target)
        decoded_df = pd.read_csv(StringIO(data))
        
        # Split the data into X and y
        X = decoded_df[selected_features]
        y = decoded_df[selected_target]
        groups = decoded_df[selected_id]
        # Define GroupKFold
        group_kfold = GroupKFold(n_splits=3)
        
        # Initialize the model
        model = xgb.XGBClassifier(n_estimators=100,
                                  max_depth=5,
                                  eval_metric='aucpr',
                                  seed=42,
                                  objective='binary:logistic',
                                  eta=0.1)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('ROC-Curve', 'Precision-Recall-Curve'))
        
        # Perform GroupKFold cross-validation
        for train_index, test_index in group_kfold.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train the model
            model.fit(X_train, y_train)

             # Make predictions
            y_pred_prob = model.predict_proba(X_test)[:,1]
            y_pred = model.predict(X_test)

            # Print classification report
            logger.info(classification_report(y_test, y_pred))

            # Calculate ROC-AUC
            roc_auc = roc_auc_score(y_test, y_pred_prob)
            logger.info(f'ROC-AUC: {roc_auc}')
            
            precision, recall, thresholds = precision_recall_curve(y_test, probas_pred=y_pred_prob, )
            #axs[1].plot(recall, precision)
            
            sensitivity, specificity, thresholds = roc_curve(y_test, y_pred_prob)
            #axs[0].plot(sensitivity, specificity)

            logger.info('sentitivity', sensitivity)
            logger.info('specificity', specificity)
            fig.add_trace(go.Scatter(x = sensitivity, y = specificity, mode = 'lines'), row = 1, col = 1)
            fig.add_trace(go.Scatter(x = precision, y = recall), row = 1, col = 2)
            logger.info('AUCPR:', auc(recall, precision))

        fig.update_layout(title = 'Model evalutaion curves')  

        logger.info("Model training completed.")
        return True, fig
    else:
        return False, {}


# Callback to show/hide the ROC curve graph based on training completion
@app.callback(Output('roc-curve-container', 'style'),
              [Input('training-completed', 'data')])
def update_roc_curve_visibility(training_completed):
    if training_completed:
        return {'display': 'block'}
    else:
        return {'display': 'none'}



if __name__ == '__main__':
    app.run_server(debug=False)
