import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict


def null_percentage(df):
    return (df.isnull().sum() / df.shape[0]) * 100


def get_most_recent_ratings(row, player_ratings_df, home_or_away_ids, numeric_stats):
    # Get match date
    match_date = row["date"]
    # Get player ids for home or away taems
    player_ids = row[home_or_away_ids].values
    # Get their ratings of players in match and that were updated before match date
    # Select the first which will be most recent cause we sort by date
    latest_ratings = (
        player_ratings_df[
            (player_ratings_df["player_api_id"].isin(player_ids))
            & (player_ratings_df["date"] < match_date)
        ]
        .groupby("player_api_id")
        .first()
        .reset_index()
    )
    # Take the mean of given stat for players
    mean_stats = latest_ratings[numeric_stats].mean()
    # Return a series (will be new columns in our dataframe)
    return pd.Series(mean_stats)


def compute_rolling_stats(df, groupby_col, value_col, window=3):
    # Compute rolling averages for given stats
    # Groupby col will be home or away and value will be a stat (ie goals)
    return (
        df.groupby(groupby_col)[value_col]
        .rolling(window=window, closed="left")
        .mean()
        .reset_index(level=0, drop=True)
    )


def expected_outcome(row):
    # Return expected team to win based on odds
    odds = {
        "home_win": row["average_home_odds"],
        "away_win": row["average_away_odds"],
        "draw": row["average_draw_odds"],
    }
    return min(odds, key=odds.get)


def cross_validation_report_classifcation(cross_val_results, print_out=False):
    accuracy = round(np.mean(cross_val_results["test_accuracy"]) * 100, 3)
    precision = round(np.mean(cross_val_results["test_precision"]) * 100, 3)
    recall = round(np.mean(cross_val_results["test_recall"]) * 100, 3)
    f1_score = round(np.mean(cross_val_results["test_f1_score"]) * 100, 3)
    if print_out:
        print(f"The accuracy of our model is {accuracy}%")
        print(f"The precision of our model is {precision}%")
        print(f"The recall of our model is {recall}%")
        print(f"The f1 score of our model is {f1_score}%")
    else:
        return accuracy, precision, recall, f1_score


def cross_val_report_lin_reg(cross_val_results, print_out=False):
    # Compute R-squared
    r2_scores = np.mean(cross_val_results["test_r2"])
    # Convert negative MSE to positive
    mse_scores = np.mean(-cross_val_results["test_neg_mse"])
    # Calculate RMSE from MSE
    rmse_scores = np.mean(np.sqrt(mse_scores))
    if print_out:
        print(f"The R-squared of our model is {r2_scores}")
        print(f"The Mean Squared Error of our model is {mse_scores}")
        print(f"The Root Mean Squared Error of our model {rmse_scores}")
    else:
        return r2_scores, mse_scores, rmse_scores


def card_stats(element):
    # Check for card_type element
    card_type_element = element.find("card_type")
    # If it exists grab the text
    card_type = card_type_element.text if card_type_element is not None else 0
    # Increment by 1 depending on card type
    return (1 if card_type == "y" else 0, 1 if card_type == "r" else 0)


def possession_stats(element):
    # Find elapsed time element
    elapsed_element = element.find("elapsed")
    # Convert to int if not none
    elapsed_time = int(elapsed_element.text) if elapsed_element is not None else 0
    # Make sure we get the end of match  stats (>= 90 min)
    if elapsed_time >= 90:
        # Get home possesion element
        homepos_element = element.find("homepos")
        home_pos = int(homepos_element.text) if homepos_element is not None else 0
        # Get away possesion element
        awaypos_element = element.find("awaypos")
        away_pos = int(awaypos_element.text) if awaypos_element is not None else 0
        # Return possesion times for both teams
        return home_pos, away_pos
    return 0, 0


def calculate_main_stats(team_id, home_team_id, away_team_id):
    # Increment stats for home and away teams
    return (1 if team_id == home_team_id else 0, 1 if team_id == away_team_id else 0)


def update_match_stats(xml_data, home_team_id, away_team_id, stat):
    # Get xml tree
    tree = ET.fromstring(xml_data)
    pos_card_stats = {"card": card_stats, "possession": possession_stats}
    home_stats = defaultdict(int)
    away_stats = defaultdict(int)
    team_id = None
    # Loop through the tree
    for element in tree.iter("value"):
        # Extract team element
        team_element = element.find("team")
        # Check it exists
        if team_element is not None:
            # Extract team id
            team_id = int(team_element.text)
        # Check if stat is card or pos
        if stat in pos_card_stats:
            # Call dictionary for card/pos funcs
            stat_result = pos_card_stats[stat](element)
            if stat == "card":
                # Grab y/r card results
                yellow_card, red_card = stat_result
                # Check team id and increment accordingly
                if team_id == home_team_id:
                    home_stats["yellow_card"] += yellow_card
                    home_stats["red_card"] += red_card
                elif team_id == away_team_id:
                    away_stats["yellow_card"] += yellow_card
                    away_stats["red_card"] += red_card
            # Do the same for possesion
            elif stat == "possession":
                home_possession, away_possession = stat_result
                home_stats["possession"] += home_possession
                away_stats["possession"] += away_possession
        # Compute main stats 
        else:
            home, away = calculate_main_stats(team_id, home_team_id, away_team_id)
            home_stats[stat] += home
            away_stats[stat] += away

    return home_stats, away_stats
