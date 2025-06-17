from flask import Flask, request, jsonify, render_template
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Initialize the Flask app
app = Flask(__name__)

# Load and preprocess the data once when the app starts
data = pd.read_csv('cricket_data.csv')
column_mapping = {
    'Batting_Runs': 'Runs',
    'Batting_SR': 'Strike Rate',
    'Bowlling_Wkts': 'Wickets',
    'Bowlling_Econ': 'Economy',
    'Opposition': 'Opponent',
    'Ground': 'Venue'
}
data.rename(columns=column_mapping, inplace=True)
data.columns = data.columns.str.strip()

# Function to recommend players
def recommend_players(selected_opponent, selected_ground):
    # --- Batsman Recommendation Logic ---
    batsman_data = data.copy()
    batsman_data['Strike Rate'] = pd.to_numeric(batsman_data['Strike Rate'].replace('DNB', pd.NA), errors='coerce')
    batsman_data['Runs'] = pd.to_numeric(batsman_data['Runs'], errors='coerce')
    batsman_data.dropna(subset=['Strike Rate', 'Runs'], inplace=True)
    batsman_data['Performance'] = batsman_data.apply(
        lambda row: 'Good' if row['Runs'] >= 50 and row['Strike Rate'] >= 100 else 'Average', axis=1
    )

    filtered_batsman_data = batsman_data[
        (batsman_data['Opponent'].str.lower().str.strip() == selected_opponent.lower()) &
        (batsman_data['Venue'].str.lower().str.strip() == selected_ground.lower())
    ]

    top_batsmen = []
    if not filtered_batsman_data.empty:
        transaction_data = pd.get_dummies(filtered_batsman_data[['Player_Name', 'Performance']])
        frequent_itemsets = apriori(transaction_data, min_support=0.1, use_colnames=True)
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
            batsmen_rules = rules[rules['consequents'].apply(lambda x: 'Performance_Good' in x or 'Performance_Average' in x)]

            batsmen_list = []
            for _, rule in batsmen_rules.iterrows():
                for item in rule['antecedents']:
                    if 'Player_Name_' in item:
                        batsmen_list.append(item.replace('Player_Name_', ''))
            top_batsmen = list(set(batsmen_list))

    # --- Bowler Recommendation Logic ---
    bowler_data = data.copy()
    bowler_data['Wickets'] = pd.to_numeric(bowler_data['Wickets'], errors='coerce')
    bowler_data['Economy'] = pd.to_numeric(bowler_data['Economy'], errors='coerce')
    bowler_data.dropna(subset=['Wickets', 'Economy'], inplace=True)
    bowler_data['Performance'] = bowler_data.apply(
        lambda row: 'Good' if row['Wickets'] >= 2 else 'Average', axis=1
    )

    filtered_bowler_data = bowler_data[
        (bowler_data['Opponent'].str.lower().str.strip() == selected_opponent.lower()) &
        (bowler_data['Venue'].str.lower().str.strip() == selected_ground.lower())
    ]

    top_bowlers = []
    if not filtered_bowler_data.empty:
        transaction_data = pd.get_dummies(filtered_bowler_data[['Player_Name', 'Performance']])
        frequent_itemsets = apriori(transaction_data, min_support=0.1, use_colnames=True)
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
            bowlers_rules = rules[rules['consequents'].apply(lambda x: 'Performance_Good' in x)]

            bowlers_list = []
            for _, rule in bowlers_rules.iterrows():
                for item in rule['antecedents']:
                    if 'Player_Name_' in item:
                        bowlers_list.append(item.replace('Player_Name_', ''))
            top_bowlers = list(set(bowlers_list))

    return {'batsmen': top_batsmen, 'bowlers': top_bowlers}

# Define the main route for the frontend
@app.route('/')
def home():
    return render_template('index.html')

# Define the API endpoint for getting recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    request_data = request.get_json()
    opponent = request_data.get('opponent', '')
    ground = request_data.get('ground', '')

    recommendations = recommend_players(opponent, ground)

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)