import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from wordcloud import WordCloud
# Import neccessary modules and libraries

# Read the dataset
zoomdf = pd.read_csv('Interaction_Data.csv')


# ---------------------- Data Cleansing ---------------------- #
# Step 1:  Remove Timestamp and Email addresse from the zoomdf dataframe.
zoomdf = zoomdf.drop(columns=['Timestamp', 'Email address'])

# Step 2: Combile all "Choose your pick" Columns into a single list column
choose_columns = [col for col in zoomdf.columns if "Choose your pick" in col]
zoomdf['Chosen Names'] = zoomdf[choose_columns].apply(lambda row: [x for x in row if pd.notnull(x)], axis=1)

# Step 3: Ensure the filler name is not in their own 'Chosen Names' list

zoomdf['Chosen Names'] = zoomdf.apply(
    lambda row: [name.strip() for name in row['Chosen Names'] if name.strip().lower() != row['Name'].strip().lower()],
    axis=1
)

# Step 4: Drop the original "Choose your pick" columns as they are now redundant
zoomdf_cleaned = zoomdf.drop(columns=choose_columns)

# print(zoomdf_cleaned.iloc[66:73]) # // Checked for a row where the filler name has all entries of his own name.

# ---------------------- Visualization and Graph Analysis ---------------------- #

# Visualization Step 1: Word Cloud of Chosen Names
# Flatten the list of all chosen names for visualization
all_chosen_names = [name for sublist in zoomdf_cleaned['Chosen Names'] for name in sublist]

# Frequency of each name
name_frequency = pd.Series(all_chosen_names).value_counts()

# 1. Bar Chart - Frequency of Chosen Names
plt.figure(figsize=(12, 6))
name_frequency.head(10).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Top 10 Most Chosen Names', fontsize=16)
plt.xlabel('Name', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2. Heatmap - Interaction Matrix
# Create an adjacency matrix
participants = zoomdf_cleaned['Name'].tolist()
matrix = pd.DataFrame(0, index=participants, columns=participants)
print(participants)
print(matrix)
for _, row in zoomdf_cleaned.iterrows():
    for chosen_name in row['Chosen Names']:  # No need for eval() as 'Chosen Names' is already a list
        if chosen_name in matrix.columns:
            matrix.loc[row['Name'], chosen_name] += 1

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, cmap='Blues', annot=False, cbar=True)
plt.title('Interaction Heatmap', fontsize=16)
plt.xlabel('Chosen Participants', fontsize=14)
plt.ylabel('Fillers', fontsize=14)
plt.tight_layout()
plt.show()




# 3. Word Cloud - Most Chosen Names
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(name_frequency)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Most Chosen Names', fontsize=16)
plt.tight_layout()
plt.show()