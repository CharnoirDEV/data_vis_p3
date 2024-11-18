import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.express as px
import base64
import io
from collections import Counter
import openpyxl 


model_path = "/model.joblib" #please update your path befoere running
model = joblib.load(model_path)


ENTITY_TAGS = ["LOC", "O", "ADDR", "POST"]
TAG_OPTIONS = ["LOC", "O", "ADDR", "POST"]


stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]


def tokens_to_features(tokens, i):
    word = tokens[i]
    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace(),
        "word.is_stopword()": word in stopwords,
        "word.isdigit()": word.isdigit(),
        "word.islen5": word.isdigit() and len(word) == 5
    }
    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace(),
            "-1.word.is_stopword()": prevword in stopwords,
            "-1.word.isdigit()": prevword.isdigit(),
        })
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace(),
            "+1.word.is_stopword()": nextword in stopwords,
            "+1.word.isdigit()": nextword.isdigit(),
        })
    else:
        features["EOS"] = True

    return features

def parse(text):
    tokens = text.split()
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    try:
        predictions = model.predict([features])[0]
    except Exception:
        predictions = ["O"] * len(tokens)  
    return tokens, predictions

def merge_error_data(existing_data, new_data):
  
    existing_data = existing_data if isinstance(existing_data, list) else []
    new_data = new_data if isinstance(new_data, list) else []

    combined_df = pd.DataFrame(existing_data + new_data)

    unexpected_tags = combined_df[~combined_df["Tag"].isin(ENTITY_TAGS)]["Tag"].unique()
    if len(unexpected_tags) > 0:
        print(f"Unexpected Tags Found in Merged Data: {unexpected_tags}")


    combined_df["Tag"] = combined_df["Tag"].apply(lambda x: x if x in ENTITY_TAGS else "UNKNOWN")

    aggregated_df = combined_df.groupby("Token").sum().reset_index()

    return aggregated_df.to_dict("records")



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = dbc.Container([
    html.H1([
        "CRF Model Token-Level Annotation and Evaluation ",
        html.I(className="bi bi-bar-chart-fill")
    ], className="text-center mb-4"),

    dcc.Store(id="history-store", storage_type="session"),
    dcc.Store(id="combined-error-data-store", storage_type="session"),

    html.Div([
    dbc.Card([
        dbc.CardBody([
            html.H3("Input Text", className="card-title text-center"),
            dbc.Row([
                dbc.Col([
                    html.Label("Input Your Phrase Here", className="fw-bold"),
                    dbc.Input(id="input-text", placeholder="Enter a phrase...", type="text", className="mb-3"),
                    dbc.Button("Detect Tokens", id="detect-tokens", color="primary", className="mb-3"),
                ]),
            ], className="mb-4"),
            html.Div(id="token-inputs", className="mb-4"),
            dbc.Button("Compare Predictions", id="compare-button", color="success", style={"display": "none"}),
            html.Div(id="comparison-table", className="mb-4"),
            dbc.Button("Show Confusion Matrix", id="show-matrix", color="info", style={"display": "none"}),
            dcc.Graph(id="confusion-matrix", style={"display": "none"}),
        ])
    ], style={"marginBottom": "20px"})
]),


    html.Div([
        html.H3("Upload File for Error Analysis", className="text-center mb-4"),
        dcc.Upload(
            id="upload-error-analysis",
            children=html.Div(["Drag and Drop or ", html.A("Select Excel File")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "10px",
                "textAlign": "center",
                "marginBottom": "10px",
                "backgroundColor": "#f8f9fa"
            },
            multiple=False
        ),
        html.Div(id="upload-analysis-message", style={"color": "green", "marginBottom": "20px", "fontStyle": "italic"}),
    ], style={"backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),


    html.Div([
        html.H3("Most Common Mistaken Tokens", className="text-center mb-4"),
        dcc.Loading(
            id="loading-error-analysis",
            type="default",
            children=dcc.Graph(id="error-analysis-graph")
        ),
    ], style={"backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),


    html.Div([
        html.H3("Top Mistaken Tokens", className="text-center mb-4"),
        html.Div(id="top-errors-table"),
    ], style={"backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

    # Pie Chart
    html.Div([
        html.H3("Mistake Proportions by Tag", className="text-center mb-4"),
        dcc.Loading(
            id="loading-pie-chart",
            type="default",
            children=dcc.Graph(id="error-pie-chart")
        ),
    ], style={"backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),
], style={"maxWidth": "900px", "margin": "auto", "paddingTop": "20px"})
    

@app.callback(
    Output("upload-analysis-message", "children"),
    Output("combined-error-data-store", "data"),
    Input("upload-error-analysis", "contents"),
    State("upload-error-analysis", "filename"),
    State("combined-error-data-store", "data")
)
def process_upload_append_data(contents, filename, combined_error_data):
    if not contents:
        return "No file uploaded.", combined_error_data


    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        if filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return "Unsupported file type. Please upload an Excel file.", combined_error_data
    except Exception as e:
        return f"Error reading file: {str(e)}", combined_error_data


    print("Initial DataFrame from uploaded file:")
    print(df.head())

    df["Tag"] = df["True Value"].apply(
        lambda x: x.strip().upper() if isinstance(x, str) and x.strip().upper() in ENTITY_TAGS else "UNKNOWN"
    )



    print("True Value Column Before Tag Assignment:")
    print(df["True Value"].unique())


    def sanitize_true_value(value):
        if not isinstance(value, str):
            return "UNKNOWN"
        clean_value = value.strip().upper()
        if clean_value not in ENTITY_TAGS:
            print(f"Unexpected True Value Found: {clean_value}")  
            return "UNKNOWN"
        return clean_value

    df["Tag"] = df["True Value"].apply(sanitize_true_value)



    df["Error"] = df["True Value"] != df["Prediction"]


    upload_error_list = (
        df[df["Error"]]
        .groupby(["Token", "Tag"])
        .size()
        .reset_index(name="Count")
        .to_dict("records")
    )

    combined_error_data = merge_error_data(combined_error_data, upload_error_list)

    for error in combined_error_data:
        if error["Tag"] not in ENTITY_TAGS:
            print(f"Unexpected Tag Found: {error['Tag']} in Token {error['Token']}")
            error["Tag"] = "UNKNOWN"

    return f"Successfully processed {len(df)} rows from {filename}.", combined_error_data


@app.callback(
    Output("token-inputs", "children"),
    Output("compare-button", "style"),
    Input("detect-tokens", "n_clicks"),
    State("input-text", "value")
)
def render_annotation_inputs(n_clicks, text):
    if not text:
        return "Please enter text to detect tokens!", {"display": "none"}


    tokens = text.split()


    inputs = [html.H5("Tokens Detected Please Fill Ground Truth:")]
    row = []  
    for i, token in enumerate(tokens):
        row.append(
            html.Div([
                html.Label(f"{token}:", style={"font-weight": "bold", "margin-right": "10px"}),
                dcc.Dropdown(
                    id={"type": "dynamic-dropdown", "index": i},
                    options=[{"label": tag, "value": tag} for tag in TAG_OPTIONS],
                    placeholder=f"Select tag for '{token}'",
                    style={"width": "150px", "display": "inline-block"}
                )
            ], style={"display": "flex", "align-items": "center", "margin-right": "20px"})
        )
        if (i + 1) % 4 == 0 or (i + 1) == len(tokens):
            inputs.append(html.Div(row, style={"display": "flex", "margin-bottom": "10px"}))
            row = []  

    return inputs, {"display": "block"}

@app.callback(
    Output("comparison-table", "children"),
    Output("show-matrix", "style"),
    Input("compare-button", "n_clicks"),
    State("input-text", "value"),
    State({"type": "dynamic-dropdown", "index": dash.ALL}, "value")
)
def compare_predictions(n_clicks, text, ground_truth_values):
    if not text or not ground_truth_values:
        return "Please provide all ground truth annotations before comparing!", {"display": "none"}

    tokens, predicted_tags = parse(text)

    if len(ground_truth_values) != len(tokens):
        return "Error: Number of annotations must match the number of tokens!", {"display": "none"}


    df = pd.DataFrame({
        "Token": tokens,
        "Ground Truth": ground_truth_values,
        "Prediction": predicted_tags
    })

    table_header = [html.Tr([html.Th("Token"), html.Th("Ground Truth"), html.Th("Prediction")])]
    table_rows = [
        html.Tr([
            html.Td(token),
            html.Td(gt, style={"color": "green" if gt == pred else "red"}),
            html.Td(pred, style={"color": "blue"})
        ])
        for token, gt, pred in zip(df["Token"], df["Ground Truth"], df["Prediction"])
    ]

    table = dbc.Table(table_header + table_rows, bordered=True, hover=True, responsive=True)

    return table, {"display": "block"}

@app.callback(
    Output("confusion-matrix", "figure"),
    Output("confusion-matrix", "style"),
    Input("show-matrix", "n_clicks"),
    State("input-text", "value"),
    State({"type": "dynamic-dropdown", "index": dash.ALL}, "value")
)
def generate_confusion_matrix(n_clicks, text, ground_truth_values):
    if not text or not ground_truth_values:
        return {}, {"display": "none"}

    tokens, predicted_tags = parse(text)

    if len(ground_truth_values) != len(tokens):
        return {}, {"display": "none"}

    cm = confusion_matrix(ground_truth_values, predicted_tags, labels=ENTITY_TAGS)
    cm_df = pd.DataFrame(cm, index=ENTITY_TAGS, columns=ENTITY_TAGS)

    fig = px.imshow(
        cm_df,
        text_auto=True,
        labels=dict(x="Predicted Tags", y="True Tags", color="Count"),
        x=ENTITY_TAGS,
        y=ENTITY_TAGS,
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )

    return fig, {"display": "block"}


@app.callback(
    Output("error-analysis-graph", "figure"),
    Input("combined-error-data-store", "data")
)
def display_error_analysis_graph(combined_error_data):
    if not combined_error_data:
        return px.bar(title="No Errors Found.")


    error_df = pd.DataFrame(combined_error_data)
 
    fig = px.bar(
        error_df,
        x="Token",
        y="Count",
        text="Count",
        title="Most Common Mistaken Tokens",
        labels={"Token": "Token", "Count": "Mistake Count"}
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

@app.callback(
    Output("top-errors-table", "children"),
    Input("combined-error-data-store", "data")
)
def highlight_top_errors(combined_error_data):
    if not combined_error_data:
        return html.Div("No errors to display.")


    error_df = pd.DataFrame(combined_error_data)
    error_df = error_df.sort_values(by="Count", ascending=False).head(20)

    table_header = [html.Tr([html.Th("Token"), html.Th("Mistake Count")])]
    table_rows = [
        html.Tr([html.Td(row["Token"]), html.Td(row["Count"])])
        for _, row in error_df.iterrows()
    ]
    table = dbc.Table(table_header + table_rows, bordered=True, hover=True, responsive=True)

    return table


@app.callback(
    Output("error-pie-chart", "figure"),
    Input("combined-error-data-store", "data")
)
def display_pie_chart(combined_error_data):
    if not combined_error_data:
        return px.pie(title="No Data Available")

    error_df = pd.DataFrame(combined_error_data)


    invalid_tags = error_df[~error_df["Tag"].isin(ENTITY_TAGS)]["Tag"].unique()
    if len(invalid_tags) > 0:
        print(f"Invalid Tags Found in Pie Chart Data: {invalid_tags}")

    error_df["Tag"] = error_df["Tag"].apply(lambda x: x if x in ENTITY_TAGS else "UNKNOWN")

    tag_summary = error_df.groupby("Tag")["Count"].sum().reset_index()


    if tag_summary.empty:
        return px.pie(title="No Data Available")

    fig = px.pie(
        tag_summary,
        names="Tag",
        values="Count",
        title="Proportion of Mistakes by Tag eg. LOC mistake X %",
        labels={"Tag": "Entity Tag", "Count": "Mistake Count"}
    )
    return fig



if __name__ == "__main__":
    app.run_server(debug=True)
