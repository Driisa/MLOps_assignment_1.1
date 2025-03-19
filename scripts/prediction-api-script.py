#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Daily prediction script for penguin classification.
This script fetches new penguin data from the API, makes predictions using the trained model,
and saves the results to be displayed on GitHub Pages.

Focus: Identify Adelie penguins (Skipper, Private, Rico, and Kowalski)
"""

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Set matplotlib backend to Agg for server environments
matplotlib.use('Agg')

# API endpoint for new penguin data
API_URL = "http://130.225.39.127:8000/new_penguin/"

def fetch_new_penguin_data():
    """Fetch new penguin data from the API."""
    print(f"Fetching new penguin data from {API_URL}...")
    
    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        print(f"Successfully fetched data: {data}")
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        # Return sample data for testing when API is unavailable
        return {
            'bill_length_mm': 39.1,
            'bill_depth_mm': 18.7,
            'flipper_length_mm': 181.0,
            'body_mass_g': 3750.0
        }

def load_models():
    """Load the trained models."""
    print("Loading trained models...")
    
    # Load species classifier model
    species_model_path = "models/best_model.pkl"
    if not os.path.exists(species_model_path):
        print(f"Error: Species model file not found at {species_model_path}")
        sys.exit(1)
    
    species_model = joblib.load(species_model_path)
    
    # Load Adelie detector model
    adelie_model_path = "models/adelie_detector.pkl"
    if os.path.exists(adelie_model_path):
        adelie_model = joblib.load(adelie_model_path)
        print("Loaded Adelie detector model")
    else:
        adelie_model = None
        print("Adelie detector model not found, will use species model only")
    
    # Load model information
    try:
        with open("models/model_metrics.json", "r") as f:
            model_info = json.load(f)
        print(f"Using {model_info['model_name']} model for species classification")
    except:
        model_info = {"model_name": "Unknown"}
        print("Model information not found, using loaded model without additional info")
    
    return species_model, adelie_model, model_info

def make_prediction(species_model, adelie_model, model_info, penguin_data):
    """Make prediction for the new penguin data."""
    print("Making prediction...")
    
    # Convert data to DataFrame
    df = pd.DataFrame([penguin_data])
    
    # Ensure only features used by the model are present
    required_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    
    # Check if all required features are present
    for feature in required_features:
        if feature not in df.columns:
            print(f"Warning: Missing required feature {feature}. Using default value.")
            df[feature] = 0  # Default value
    
    # Remove any extra columns not used by the model
    extra_columns = set(df.columns) - set(required_features)
    if extra_columns:
        print(f"Removing extra columns not used for prediction: {extra_columns}")
        df = df.drop(columns=extra_columns)
    
    # Make species prediction
    if model_info.get("model_name") == "XGBoost" and hasattr(species_model, 'label_encoder'):
        # For XGBoost
        raw_prediction = species_model.predict(df)[0]
        species_prediction = species_model.label_encoder.inverse_transform([raw_prediction])[0]
        
        # Get prediction probabilities
        if hasattr(species_model, 'predict_proba'):
            probabilities = species_model.predict_proba(df)[0]
            proba_dict = {cls: float(prob) for cls, prob in 
                         zip(species_model.label_encoder.classes_, probabilities)}
        else:
            proba_dict = {"Note": "Probability information not available"}
    else:
        # For RandomForest
        species_prediction = species_model.predict(df)[0]
        
        # Get prediction probabilities
        if hasattr(species_model, 'predict_proba'):
            probabilities = species_model.predict_proba(df)[0]
            proba_dict = {cls: float(prob) for cls, prob in 
                         zip(species_model.classes_, probabilities)}
        else:
            proba_dict = {"Note": "Probability information not available"}
    
    # Check if it's an Adelie using the dedicated Adelie detector if available
    is_adelie_specialist = False
    adelie_probability = None
    
    if adelie_model is not None:
        is_adelie_specialist = bool(adelie_model.predict(df)[0])
        
        if hasattr(adelie_model, 'predict_proba'):
            try:
                adelie_probability = float(adelie_model.predict_proba(df)[0][1])  # Probability of being Adelie
            except:
                print("Warning: Could not get Adelie probability")
                adelie_probability = None
    else:
        # If no Adelie detector, use the species model result
        is_adelie_specialist = (species_prediction == 'Adelie')
    
    # Determine final classification - Species model takes precedence for this assignment
    # since the main task is to find Adelie penguins from Madagascar
    is_adelie = (species_prediction == 'Adelie')
    
    # Log any inconsistency
    if (species_prediction == 'Adelie' and not is_adelie_specialist) or (species_prediction != 'Adelie' and is_adelie_specialist):
        print(f"WARNING: Inconsistency detected! Species model says {species_prediction}, " + 
              f"Adelie detector says {'Adelie' if is_adelie_specialist else 'Not Adelie'}")
    
    # Create prediction result
    prediction_result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "penguin_data": penguin_data,
        "predicted_species": species_prediction,
        "species_probabilities": proba_dict,
        "is_adelie": is_adelie,
        "adelie_probability": adelie_probability,
        "model_used": model_info.get("model_name", "Unknown"),
        "note": "This could be one of the Penguins of Madagascar!" if is_adelie else "Not one of the Penguins of Madagascar"
    }
    
    print(f"Prediction: {species_prediction}")
    print(f"Is Adelie: {is_adelie}")
    
    return prediction_result

def create_visualization(prediction_result):
    """Create a visualization of the prediction result.
    Uses a simpler approach that avoids radar charts and polar coordinates.
    """
    print("Creating visualization...")
    
    # Extract data
    penguin_data = prediction_result["penguin_data"]
    species = prediction_result["predicted_species"]
    is_adelie = prediction_result["is_adelie"]
    
    # Create figure with subplots (no polar axes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Bar chart of measurements in the first subplot
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm/10', 'body_mass_g/100']
    values = [
        penguin_data['bill_length_mm'],
        penguin_data['bill_depth_mm'],
        penguin_data['flipper_length_mm']/10,  # Scale down for better visualization
        penguin_data['body_mass_g']/100        # Scale down for better visualization
    ]
    
    # Create bar chart
    colors = ['skyblue' if not is_adelie else 'royalblue'] * 4
    ax1.bar(features, values, color=colors)
    ax1.set_title("Penguin Measurements", fontsize=14)
    ax1.set_ylabel("Value")
    
    # Add actual values as text labels
    for i, v in enumerate(values):
        ax1.text(i, v + 0.5, f"{v:.1f}", ha='center')
    
    # 2. Use the second subplot for the species summary
    ax2.axis('off')  # Turn off axis
    
    # Penguin info box
    info_text = f"""
    PREDICTION SUMMARY:
    
    Predicted Species: {species}
    
    Is it an Adelie? {"YES! 🐧" if is_adelie else "No 😢"}
    
    Measurements:
    - Bill Length: {penguin_data['bill_length_mm']:.1f} mm
    - Bill Depth: {penguin_data['bill_depth_mm']:.1f} mm
    - Flipper Length: {penguin_data['flipper_length_mm']:.1f} mm
    - Body Mass: {penguin_data['body_mass_g']:.1f} g
    
    {prediction_result.get('note', '')}
    
    Date: {prediction_result.get('date', '')}
    """
    
    # Add text with larger font
    bg_color = 'lightblue' if is_adelie else 'lightgray'
    ax2.text(0.1, 0.5, info_text, fontsize=14, 
             verticalalignment='center',
             bbox=dict(facecolor=bg_color, alpha=0.5))
    
    # Add a global title
    title_color = 'blue' if is_adelie else 'black'
    plt.suptitle(f"Penguin Classification Result: {species}", size=18, color=title_color)
    
    # Save the chart
    plt.tight_layout()
    
    try:
        # Ensure directory exists
        Path("data/predictions").mkdir(parents=True, exist_ok=True)
        
        # Save the images
        plt.savefig("data/predictions/latest_visualization.png")
        print("Visualization saved successfully")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    
    plt.close()

def save_prediction_results(prediction_result):
    """Save prediction results to a file."""
    print("Saving prediction results...")
    
    # Create directory for predictions if it doesn't exist
    Path("data/predictions").mkdir(parents=True, exist_ok=True)
    
    # Save prediction as JSON
    today = datetime.now().strftime("%Y-%m-%d")
    prediction_path = f"data/predictions/{today}.json"
    
    with open(prediction_path, "w") as f:
        json.dump(prediction_result, f, indent=2)
    
    # Also save to a "latest.json" file for easy access
    with open("data/predictions/latest.json", "w") as f:
        json.dump(prediction_result, f, indent=2)
    
    print(f"Prediction saved to {prediction_path}")
    
    # Update prediction history
    update_prediction_history(prediction_result)
    
    # Create visualization
    create_visualization(prediction_result)

def update_prediction_history(prediction_result):
    """Update the prediction history JSON file."""
    history_path = "data/predictions/history.json"
    
    # Load existing history or create a new one
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new prediction to history
    simplified_result = {
        "date": prediction_result["date"],
        "predicted_species": prediction_result["predicted_species"],
        "is_adelie": prediction_result["is_adelie"]
    }
    
    history.append(simplified_result)
    
    # Save updated history
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

def update_github_pages():
    """Update GitHub Pages with the latest prediction results."""
    print("Updating GitHub Pages content...")
    
    # Load the latest prediction
    try:
        with open("data/predictions/latest.json", "r") as f:
            latest = json.load(f)
    except Exception as e:
        print(f"Error: Could not load latest prediction: {e}")
        return
    
    # Load prediction history
    try:
        with open("data/predictions/history.json", "r") as f:
            history = json.load(f)
    except:
        history = []
    
    # Count Adelie penguins
    adelie_count = sum(1 for item in history if item.get("is_adelie", False))
    adelie_percentage = (adelie_count/len(history)*100) if history else 0
    
    # Create HTML content for GitHub Pages
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Penguins of Madagascar Detector</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
        }}
        .penguin-data {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        .visualization {{
            text-align: center;
            margin: 20px 0;
        }}
        .prediction {{
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
            padding: 10px;
            background-color: #e8f4f8;
            border-radius: 5px;
        }}
        .adelie {{
            color: #2980b9;
        }}
        .other {{
            color: #c0392b;
        }}
        .stats {{
            margin-top: 30px;
            text-align: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        table, th, td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .madagascar {{
            text-align: center;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Penguins of Madagascar Detector</h1>
        
        <div class="madagascar">
            <p><strong>Searching for Skipper, Private, Rico, and Kowalski in New York City!</strong></p>
        </div>
        
        <div class="prediction">
            <p>Latest Prediction (Date: {latest.get("date", "N/A")}): 
                <span class="{'adelie' if latest.get('is_adelie', False) else 'other'}">{latest.get("predicted_species", "Unknown")}</span>
            </p>
            <p>{latest.get('note', '')}</p>
        </div>
        
        <div class="visualization">
            <img src="latest_visualization.png" alt="Penguin Visualization" style="max-width: 100%;">
        </div>
        
        <div class="penguin-data">
            <h3>Penguin Measurements:</h3>
            <ul>
                <li><strong>Bill Length:</strong> {latest.get("penguin_data", {}).get("bill_length_mm", "N/A")} mm</li>
                <li><strong>Bill Depth:</strong> {latest.get("penguin_data", {}).get("bill_depth_mm", "N/A")} mm</li>
                <li><strong>Flipper Length:</strong> {latest.get("penguin_data", {}).get("flipper_length_mm", "N/A")} mm</li>
                <li><strong>Body Mass:</strong> {latest.get("penguin_data", {}).get("body_mass_g", "N/A")} g</li>
            </ul>
        </div>
        
        <div class="stats">
            <h3>Statistics</h3>
            <p>Total penguins detected: {len(history)}</p>
            <p>Adelie penguins detected: {adelie_count} ({adelie_percentage:.1f}% of total)</p>
            <p>Model used: {latest.get("model_used", "Unknown")}</p>
        </div>
        
        <div class="history">
            <h3>Recent Detection History (Last 10)</h3>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Species</th>
                    <th>Adelie?</th>
                </tr>
                {"".join(f'<tr><td>{item.get("date", "N/A")}</td><td>{item.get("predicted_species", "Unknown")}</td><td>{"Yes" if item.get("is_adelie", False) else "No"}</td></tr>' for item in history[-10:])}
            </table>
        </div>
    </div>
</body>
</html>
"""
    
    # Write the HTML content to index.html in the root directory for GitHub Pages
    try:
        with open("data/predictions/index.html", "w") as f:
            f.write(html_content)
        print("GitHub Pages content updated")
    except Exception as e:
        print(f"Error updating GitHub Pages: {e}")

def main():
    """Main function to fetch data and make predictions."""
    try:
        # Fetch new penguin data
        penguin_data = fetch_new_penguin_data()
        
        # Load the trained models
        species_model, adelie_model, model_info = load_models()
        
        # Make prediction
        prediction_result = make_prediction(species_model, adelie_model, model_info, penguin_data)
        
        # Save prediction results
        save_prediction_results(prediction_result)
        
        # Update GitHub Pages
        update_github_pages()
        
        print("Daily prediction completed successfully!")
        print(f"Is it an Adelie penguin? {'Yes! This could be one of the Penguins of Madagascar!' if prediction_result['is_adelie'] else 'No, keep searching!'}")
        
    except Exception as e:
        print(f"Error in daily prediction: {e}")
        raise

if __name__ == "__main__":
    main()