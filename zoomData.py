# ---------------------- Import Necessary Libraries ---------------------- #
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from wordcloud import WordCloud
from networkx.algorithms.community import greedy_modularity_communities
from textblob import TextBlob  # For sentiment analysis
print('Setup Complete')

# ---------------------- Load and Clean Data ---------------------- #

# Read the dataset
zoomdf = pd.read_csv('Interaction_Data.csv')
#print(zoomdf.size)
#print(zoomdf.head())


# ---------------------- Load and Clean Data ---------------------- #

# Step 1: Load dataset
zoomdf = pd.read_csv('Interaction_Data.csv')

# Step 2: Drop unnecessary columns
zoomdf = zoomdf.drop(columns=['Timestamp', 'Email address'])

# Step 3: Combine all "Choose your pick" columns into a single list column
choose_columns = [col for col in zoomdf.columns if "Choose your pick" in col]
zoomdf['Chosen Names'] = zoomdf[choose_columns].apply(lambda row: [x for x in row if pd.notnull(x)], axis=1)

# Step 4: Remove filler names from their own 'Chosen Names' list
zoomdf['Chosen Names'] = zoomdf.apply(
    lambda row: [name.strip() for name in row['Chosen Names'] if name.strip().lower() != row['Name'].strip().lower()],
    axis=1
)

# Step 5: Remove duplicates within each 'Chosen Names' list
zoomdf['Chosen Names'] = zoomdf['Chosen Names'].apply(lambda names: list(set(names)))

# Step 6: Drop the original "Choose your pick" columns
zoomdf_cleaned = zoomdf.drop(columns=choose_columns)

#print(zoomdf_cleaned.iloc[66:73]) # // Checked for a row where the filler name has all entries of his own name.


# ---------------------- Visualization and Graph Analysis ---------------------- #

# Flatten the list of all chosen names for visualization
all_chosen_names = [name for sublist in zoomdf_cleaned['Chosen Names'] for name in sublist]

# Frequency of each name
name_frequency = pd.Series(all_chosen_names).value_counts()
#print(name_frequency)

# Function to create gradient bars
def gradient_bar(ax, x, height, width=0.8, color_map='viridis'):
    cmap = plt.get_cmap(color_map)
    for i in range(len(height)):
        bar = ax.bar(x[i], height[i], width, color=cmap(i / len(height)), edgecolor='black')

# 1. Bar Chart - Frequency of Chosen Names

plt.figure(figsize=(12, 6))
ax = plt.gca()  # Get current axes
x_labels = name_frequency.head(10).index
y_values = name_frequency.head(10).values
gradient_bar(ax, x_labels, y_values)
plt.title('Top 10 Most Chosen Names', fontsize=16)
plt.xlabel('Name', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2. Word Cloud - Most Chosen Names
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(name_frequency)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Most Chosen Names', fontsize=16)
plt.tight_layout()
plt.show()


# 3. Heatmap - Interaction Matrix
# Create an adjacency matrix
participants = zoomdf_cleaned['Name'].tolist()
matrix = pd.DataFrame(0, index=participants, columns=participants)
# print(participants)
# print(matrix)
for _, row in zoomdf_cleaned.iterrows():
    for chosen_name in row['Chosen Names']:  
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


# Graph Analysis 
#Step 1: Build the Interaction Network Graph
# Create a directed graph using NetworkX
G = nx.DiGraph()

# Add edges to the graph (from filler name to chosen names)
for _, row in zoomdf_cleaned.iterrows():
    filler = row['Name']
    chosen_names = row['Chosen Names']
    for chosen_name in chosen_names:
        G.add_edge(filler, chosen_name)

# Graph Analysis Step 2: Visualize the Interaction Network
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)  # Positioning nodes
nx.draw(
    G, pos, with_labels=True, node_color='skyblue', node_size=2000,
    edge_color='gray', font_size=8, font_weight='bold', arrowsize=20
)
plt.title('Interaction Network Graph', fontsize=16)
plt.show()

# Graph Analysis Step 3: Degree Centrality Analysis
degree_centrality = nx.degree_centrality(G)
sorted_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print(sorted_centrality)

# Display the top 10 participants with the highest degree centrality
print("Top 10 Participants by Degree Centrality:")
for name, centrality in sorted_centrality:
    print(f"{name}: {centrality:.4f}")

# Graph Analysis Step 4: Community Detection (Using Louvain method for modularity-based communities)

communities = greedy_modularity_communities(G.to_undirected())  # Use undirected graph for community detection
print("\nCommunities Detected:")
for i, community in enumerate(communities):
    print(f"Community {i+1}: {list(community)}")


# Visualize communities with color coding
color_map = {}
for i, community in enumerate(communities):
    for name in community:
        color_map[name] = i

node_colors = [color_map[node] for node in G.nodes]
plt.figure(figsize=(12, 12))
nx.draw(
    G, pos, with_labels=True, node_color=node_colors, node_size=2000,
    edge_color='gray', font_size=8, font_weight='bold', arrowsize=20,
    cmap=plt.cm.tab20  # Use a colormap for coloring
)
plt.title('Interaction Network with Communities', fontsize=16)
plt.show()

# ---------------------- Participant Interaction Metrics ---------------------- #

# Step 1: Interaction Frequency
# Calculate the total number of interactions for each participant
interaction_frequency = matrix.sum(axis=1)  # Total interactions per filler (row-wise sum)
print("Interaction Frequency for each participant:")
print(interaction_frequency)

# Step 2: Top Interactors
# Combine incoming and outgoing interactions for ranking
total_interactions = interaction_frequency + matrix.sum(axis=0)  # Add outgoing and incoming
top_interactors = total_interactions.sort_values(ascending=False).head(10)
print("\nTop 10 Interactors:")
print(top_interactors)

# Step 3: Heatmap - Interaction Frequency
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, cmap='coolwarm', annot=True, fmt='d', cbar=True)
plt.title('Participant Interaction Frequency Heatmap', fontsize=16)
plt.xlabel('Chosen Participants', fontsize=14)
plt.ylabel('Fillers', fontsize=14)
plt.tight_layout()
plt.show()
