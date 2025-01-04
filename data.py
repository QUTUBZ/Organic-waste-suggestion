# Saving the generated dataset to a CSV file
import pandas as pd
import numpy as np

# Number of samples to generate (500 rows)
num_samples = 500

# List of plant names
plant_names = [
    "Tomato", "Spinach", "Cabbage", "Watermelon", "Onion", "Mango", "Beetroot", "Carrot", "Peas", "Lettuce",
    "Potato", "Cauliflower", "Brinjal", "Cucumber", "Chili", "Garlic", "Pumpkin", "Radish", "Broccoli", "Zucchini"
]

# List of agricultural waste products
agriculture_products = [
    "Banana Peels", "Rice Husk", "Corn Stalks", "Sugarcane Bagasse", "Coconut Shells",
    "Chicken Manure", "Cow Manure", "Compost", "Coffee Grounds", "Grass Clippings",
    "Vegetable Scraps", "Fruit Scraps", "Seaweed Extract", "Worm Castings", "Alfalfa Hay",
    "Mulch (Wood Chips)", "Tree Leaves", "Pea Pods", "Eggshells", "Fish Emulsion"
]

# Randomly generate data for the features
np.random.seed(42)

# Random data for features
soil_types = ['Loamy', 'Sandy', 'Clay']
temperature = np.random.randint(15, 35, size=num_samples)  # Temperature between 15°C and 35°C
moisture = np.random.randint(30, 80, size=num_samples)  # Moisture content between 30% and 80%
rainfall = np.random.randint(0, 300, size=num_samples)  # Rainfall between 0 mm and 300 mm
growth_time = np.random.randint(30, 180, size=num_samples)  # Growth time between 30 and 180 days
soil_type = np.random.choice(soil_types, size=num_samples)
plant_name = np.random.choice(plant_names, size=num_samples)

# Define a function to determine the growth stage based on the attributes
def determine_growth_stage(row):
    if row['growth_time'] <= 60 and row['temperature'] >= 20 and row['moisture'] >= 40:
        return 'Initial Stage'
    elif row['growth_time'] <= 120 and row['temperature'] >= 20 and row['rainfall'] >= 50:
        return 'Growth Stage'
    else:
        return 'Final Stage'

# Apply function to generate growth stages
growth_stages = [determine_growth_stage(row) for _, row in pd.DataFrame({
    'temperature': temperature,
    'moisture': moisture,
    'rainfall': rainfall,
    'growth_time': growth_time
}).iterrows()] 

# Generate compost recommendations and compost composition
compost_recommendations = []
compost_compositions = []

for stage, soil in zip(growth_stages, soil_type):
    # Select 3 random agricultural products based on stage and soil
    selected_products = np.random.choice(agriculture_products, size=3, replace=False)
    compost_recommendations.append(", ".join(selected_products))
    
    # Compost composition based on stage
    if stage == 'Initial Stage':
        compost_compositions.append({'C:N Ratio': '25:1', 'Moisture Content': '60%', 'Nitrogen Content': '1.5%', 'pH Level': '6.0'})
    elif stage == 'Growth Stage':
        compost_compositions.append({'C:N Ratio': '30:1', 'Moisture Content': '55%', 'Nitrogen Content': '2.0%', 'pH Level': '6.2'})
    else:
        compost_compositions.append({'C:N Ratio': '35:1', 'Moisture Content': '50%', 'Nitrogen Content': '2.5%', 'pH Level': '6.5'})

# Create DataFrame
df = pd.DataFrame({
    'Plant Name': plant_name,
    'Soil Type': soil_type,
    'Temperature': temperature,
    'Moisture': moisture,
    'Rainfall': rainfall,
    'Growth Time': growth_time,
    'Growth Stage': growth_stages,
    'Compost Waste Recommendations': compost_recommendations,
    'Compost C:N Ratio': [comp['C:N Ratio'] for comp in compost_compositions],
    'Compost Moisture Content': [comp['Moisture Content'] for comp in compost_compositions],
    'Compost Nitrogen Content': [comp['Nitrogen Content'] for comp in compost_compositions],
    'Compost pH Level': [comp['pH Level'] for comp in compost_compositions],
})

# Saving the dataset to a CSV file
file_path = "agriculture_waste_recommendations.csv"
df.to_csv(file_path, index=False)

file_path
