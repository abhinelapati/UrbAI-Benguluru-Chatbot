import nltk
from fuzzywuzzy import fuzz

# Function to normalize text for matching (duplicated for independence)
def normalize_text(text):
    text = text.lower()
    replacements = {
        "st": "street",
        "ave": "avenue",
        "rd": "road",
        "blvd": "boulevard"
    }
    tokens = nltk.word_tokenize(text)
    normalized_tokens = [replacements.get(token, token) for token in tokens]
    return " ".join(normalized_tokens)

# Function to retrieve real-time data (simplified version for commute needs)
def get_realtime_data_for_commute(query_type, query, realtime_data):
    query_type_lower = query_type.lower()
    if query_type_lower not in realtime_data:
        return None
    
    data = realtime_data.get(query_type_lower, {})
    normalized_query = normalize_text(query)
    
    for key, value in data.items():
        normalized_key = normalize_text(key)
        similarity = fuzz.partial_ratio(normalized_key, normalized_query)
        if similarity >= 80:
            if query_type_lower == "traffic":
                return f"Traffic on {key} is {value['status']}"
            elif query_type_lower == "transit":
                return f"{key} is {value['status']} with a delay of {value['delay']}"
    return None

# Main function to calculate eco-friendly commute recommendations
def calculate_commute_recommendations(start, destination, priority, realtime_data):
    # Define routes with basic attributes
    routes = [
        {"mode": "Car", "route": f"{start} to {destination} via MG Road", "distance_km": 10, "time_min": 30, "cost_inr": 100, "carbon_g_km": 150},
        {"mode": "BMTC Bus", "route": f"{start} to {destination} via Route 500", "distance_km": 12, "time_min": 45, "cost_inr": 30, "carbon_g_km": 50},
        {"mode": "Namma Metro", "route": f"{start} to {destination} via Purple Line", "distance_km": 11, "time_min": 35, "cost_inr": 40, "carbon_g_km": 30},
        {"mode": "Cycling", "route": f"{start} to {destination} via Cubbon Park", "distance_km": 9, "time_min": 50, "cost_inr": 0, "carbon_g_km": 0}
    ]

    # Adjust routes based on real-time data
    for route in routes:
        if route["mode"] == "Car":
            traffic_info = get_realtime_data_for_commute("traffic", "MG Road", realtime_data)
            if traffic_info and "heavy" in traffic_info.lower():
                route["time_min"] += 15
        elif route["mode"] == "Namma Metro":
            transit_info = get_realtime_data_for_commute("transit", "Metro Purple Line", realtime_data)
            if transit_info and "delayed" in transit_info.lower():
                route["time_min"] += 10

    # Calculate total carbon emissions
    for route in routes:
        route["total_carbon_g"] = route["distance_km"] * route["carbon_g_km"]

    # Normalize scores
    max_time = max(route["time_min"] for route in routes)
    max_cost = max(route["cost_inr"] for route in routes)
    max_carbon = max(route["total_carbon_g"] for route in routes) or 1

    # Assign weights based on priority
    if priority == "Fastest":
        weights = {"time": 0.5, "cost": 0.2, "carbon": 0.3}
    elif priority == "Cheapest":
        weights = {"time": 0.2, "cost": 0.5, "carbon": 0.3}
    else:  # Greenest
        weights = {"time": 0.2, "cost": 0.3, "carbon": 0.5}

    # Calculate scores for each route
    for route in routes:
        time_score = 1 - (route["time_min"] / max_time) if max_time > 0 else 0
        cost_score = 1 - (route["cost_inr"] / max_cost) if max_cost > 0 else 0
        carbon_score = 1 - (route["total_carbon_g"] / max_carbon) if max_carbon > 0 else 1
        route["score"] = (time_score * weights["time"] + cost_score * weights["cost"] + carbon_score * weights["carbon"]) * 100

    # Sort routes by score and return top 3
    routes.sort(key=lambda x: x["score"], reverse=True)
    return routes[:3]