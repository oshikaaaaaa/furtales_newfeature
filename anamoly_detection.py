import pandas as pd

def detect_anomalies_detailed(new_data, model, model_features, label_encoders):
    """
    Detect anomalies in new cat health data and provide detailed insights.
    Uses the saved label encoders for consistent interpretation.
    """
    # Ensure the new data has the same features in the same order as training
    test_data = pd.DataFrame(index=new_data.index)
    for feature in model_features:
        if feature in new_data.columns:
            test_data[feature] = new_data[feature]
        else:
            # If a feature is missing, fill with a placeholder
            test_data[feature] = 0

    # Handle missing values
    test_data = test_data.fillna(0)  # Simple imputation for demonstration

    # Make prediction (-1 for anomalies, 1 for normal)
    prediction = model.predict(test_data)

    # Get anomaly score (negative values indicate anomalies)
    anomaly_score = model.decision_function(test_data)

    results = {
        "is_anomaly": prediction[0] == -1,
        "anomaly_score": anomaly_score[0],
        "insights": []
    }

    # Generate insights only if it's an anomaly
    if results["is_anomaly"]:
        # Calculate feature importance by perturbation
        feature_contributions = {}
        baseline_score = anomaly_score[0]

        for feature in model_features:
            # Create a copy with this feature set to a neutral value
            perturbed_data = test_data.copy()
            perturbed_data[feature] = 0  # Using 0 as placeholder

            # Check how anomaly score changes
            new_score = model.decision_function(perturbed_data)[0]

            # If score improves (becomes more positive), this feature contributes to anomaly
            contribution = new_score - baseline_score
            feature_contributions[feature] = contribution

        # Sort features by contribution to anomaly
        sorted_contributions = sorted(
            feature_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Generate insights based on top contributing features
        for feature, contribution in sorted_contributions[:3]:
            if contribution > 0:  # Only include significant contributions
                # Translate encoded values to human-readable insights using label encoders
                if feature == "sleep_estimate_encoded":
                    encoded_value = int(test_data[feature].iloc[0])
                    sleep_value = label_encoders['sleep_estimate'].inverse_transform([encoded_value])[0]
                    results["insights"].append(f"Unusual sleep pattern: {sleep_value}")

                elif feature == "food_midpoint":
                    food_value = test_data[feature].iloc[0]
                    if food_value < 100:
                        results["insights"].append("Unusually low food intake")
                    elif food_value > 160:
                        results["insights"].append("Unusually high food intake")
                    else:
                        results["insights"].append("Unusual change in food intake pattern")

                elif feature == "mood_encoded":
                    encoded_value = int(test_data[feature].iloc[0])
                    mood_value = label_encoders['mood'].inverse_transform([encoded_value])[0]
                    results["insights"].append(f"Unusual mood: {mood_value}")

                elif feature == "activity_level_encoded":
                    encoded_value = int(test_data[feature].iloc[0])
                    activity_value = label_encoders['activity_level'].inverse_transform([encoded_value])[0]
                    results["insights"].append(f"Unusual activity level: {activity_value}")

                elif feature == "vocalization_level_encoded":
                    encoded_value = int(test_data[feature].iloc[0])
                    vocal_value = label_encoders['vocalization_level'].inverse_transform([encoded_value])[0]
                    results["insights"].append(f"Unusual vocalization: {vocal_value}")

                elif feature == "affection_level_encoded":
                    encoded_value = int(test_data[feature].iloc[0])
                    affection_value = label_encoders['affection_level'].inverse_transform([encoded_value])[0]
                    results["insights"].append(f"Unusual affection level: {affection_value}")

                elif feature == "unusual_behavior":
                    if test_data[feature].iloc[0] == 1:
                        results["insights"].append("Explicit unusual behavior reported")

                elif feature == "visible_issues_flag":
                    if test_data[feature].iloc[0] == 1:
                        results["insights"].append("Visible health issues noted")

        # Add additional contextual insights
        if not results["insights"]:
            results["insights"].append("Unusual pattern detected but no single clear cause")

        # Add alert level based on anomaly score
        if anomaly_score[0] < -0.3:
            results["alert_level"] = "High"
            results["recommendation"] = "Consider consulting a veterinarian"
        elif anomaly_score[0] < -0.15:
            results["alert_level"] = "Medium"
            results["recommendation"] = "Monitor closely over the next few days"
        else:
            results["alert_level"] = "Low"
            results["recommendation"] = "Keep an eye on your cat's behavior"

    else:
        results["insights"].append("No unusual patterns detected")
        results["alert_level"] = "None"
        results["recommendation"] = "Continue regular care"

    return results


# Step 6: Function to analyze new log entries with consistent encoding
def analyze_new_cat_log(new_log_entry, model, model_features, label_encoders):
    """
    Analyze a new cat log entry for anomalies.
    Handles preprocessing using the same label encoders from training.
    """
    # Preprocess the raw entry to match training format
    processed = {}

    # Process food range
    if 'food_range' in new_log_entry:
        food_str = new_log_entry['food_range']
        if '<' in food_str:
            processed['food_midpoint'] = 90
        elif '>' in food_str:
            processed['food_midpoint'] = 190
        elif '-' in food_str:
            # Remove 'g' and extract low and high values
            food_str = food_str.replace('g', '')
            low, high = map(int, food_str.split('-'))
            processed['food_midpoint'] = (low + high) / 2

    # Process categorical variables using the same label encoders from training
    categorical_cols = ['sleep_estimate', 'mood', 'activity_level', 'vocalization_level', 'affection_level']

    for col in categorical_cols:
        if col in new_log_entry and col in label_encoders:
            le = label_encoders[col]
            # Handle unseen categories safely
            try:
                processed[col + '_encoded'] = le.transform([new_log_entry[col]])[0]
            except ValueError:
                # If the category wasn't seen during training, use the most common category
                processed[col + '_encoded'] = 0  # Default to the first category
                print(f"Warning: '{new_log_entry[col]}' not found in training data for '{col}'")

    # Process boolean fields
    if 'unusual_behavior' in new_log_entry:
        processed['unusual_behavior'] = 1 if new_log_entry['unusual_behavior'] else 0

    if 'visible_issues' in new_log_entry:
        issues = new_log_entry['visible_issues']
        processed['visible_issues_flag'] = 0 if issues == '[]' or issues == [] else 1

    # Create DataFrame from processed entry
    processed_df = pd.DataFrame([processed])

    # Detect anomalies using the detailed function
    results = detect_anomalies_detailed(processed_df, model, model_features, label_encoders)

    # Add original entry info for reference
    results['original_entry'] = new_log_entry

    return results