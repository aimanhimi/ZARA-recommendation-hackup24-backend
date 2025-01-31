import pandas as pd
from utils import extract_info_from_url
# Load the dataset
df = pd.read_csv('inditextech_hackupc_challenge_images.csv')

# Apply the function to each URL in the IMAGE_VERSION_3 column
df[['season', 'demographic']] = df['IMAGE_VERSION_3'].apply(extract_info_from_url).apply(pd.Series)

# Drop rows where either 'season' or 'demographic' could not be determined
df.dropna(subset=['season', 'demographic'], inplace=True)

# Function to retrieve data for specific group
def get_grouped_data(df, season, demographic):
    # Filter the DataFrame based on season and demographic, and select only the 'IMAGE_VERSION_3' column
    return df[(df['season'] == season) & (df['demographic'] == demographic)]['IMAGE_VERSION_3']

# Retrieve groups
group_v0 = get_grouped_data(df, 'V', '1')  # Women's summer clothing
group_v1 = get_grouped_data(df, 'V', '2')  # Men's summer clothing
group_v2 = get_grouped_data(df, 'V', '3')  # Kids' summer clothing
group_i0 = get_grouped_data(df, 'W', '1')  # Women's winter clothing
group_i1 = get_grouped_data(df, 'W', '2')  # Men's winter clothing
group_i2 = get_grouped_data(df, 'W', '3')  # Kids' winter clothing

# Save the grouped data to CSV files
group_v0.to_csv('women_summer_clothing.csv', index=False)
group_v1.to_csv('men_summer_clothing.csv', index=False)
group_v2.to_csv('kids_summer_clothing.csv', index=False)
group_i0.to_csv('women_winter_clothing.csv', index=False)
group_i1.to_csv('men_winter_clothing.csv', index=False)
group_i2.to_csv('kids_winter_clothing.csv', index=False)

print("Datasets have been successfully grouped and saved.")
