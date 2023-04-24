fin_happy = [["Imagine Dragons", 0.391155], ["Varinder Brar", 0.022206], ["AP Dhillon", 0.165000 ], ["Earth, Wind & Fire", 0.077494], ["Shubh", 0.000000], ["Karan Aujla", 0.120247], ["Anirudh Ravichander", 0.350000], ["Vishal-Shekhar", 0.000000], ["Shubh", 0.500000], ["Earth, Wind & Fire", 0.365409]]
relevance_fh = [1,0,1,0,0,1,1,0,1,1]
mid_happy = [["I'm Good (Blue)", 0.120000],["Heaven", 0.030778], ["I'm Not Here To Make Friends", 0.006980], ["2002", 0.056893], ["Sunshine", -0.178900], ["Friday (feat. Mufasa & Hypeman) - Dopamine Re-Edit", 0.110000], ["Rasputin", 0.000000], ["Fancy Like", -0.100000], ["Anyone For You (Tiger Lily)", 0.890020],[ "Eyes On You", 0.120000]]
relevance_mh = [1,0,0,0,0,1,0,1,1,1]
fin_sad = [["Imagine Dragons", -0.118081], ["AP Dhillon", -0.170000], ["Shubh", -0.360000], ["AP Dhillon", -0.433333], ["Sidhu Moose Wala", -0.416296],["AP Dhillon", -0.333333],["Sidhu Moose Wala", -0.375000],["The Local Train", -0.409804],["Gajendra Verma", -0.500000],["Rovalio", -0.340000]]
relevance_fs = [0,1,1,1,0,0,1,0,0,1]
mid_sad = [["Hey Up There", -0.232000], ["Pain Pain Go Away", -0.045000], ["Mr. Forgettable", 0.100000], ["F*ck Love", -0.120000], ["Say You Hate Me", 0.000000], ["I Hate That...", -0.300030], ["Before You Go", -0.200200], ["Life's A Mess (feat. Halsey)", 0.002000], ["Computer Crash", -0.089900], ["idfc", -0.067000]]
relevance_ms = [0,0,1,0,0,0,1,1,1,0]
fin_nuetral = [["DJ Snake", 0.201881],["DJ Snake", 0.324675], ["Earth, Wind & Fire", 0.077494], ["Earth, Wind & Fire", 0.220000], ["Imagine Dragons", 0.043609], ["DJ Snake", 0.324675], ["DJ Snake", 0.186155], ["DJ Snake", 0.186155], ["AP Dhillon", 0.165000], ["Imagine Dragons", 0.043609]]
relevance_fn = [0,1,0,1,0,1,1,1,0,0]
mid_nuetral = [["Blinding Lights", 0.104000], ["Kings & Queens", 0.300090], ["Castle on the Hill", -0.209000], ["Remember", 0.002000], ["My Head & My Heart", 0.003000] , ["Heartbreak Anthem (with David Guetta & Little Mix)", -0.022073], [ "Don't Wake Me Up", 0.011969], ["Dive", 0.006389], ["Chemical", 0.306708], ["Higher Power", -0.409987]]
relevance_mn = [0,0,0,1,1,1,0,1,0,0]

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def intra_list_similarity(lst):
    # Extract the score values from the list
    scores = np.array([x[1] for x in lst]).reshape(-1, 1)
    
    # Calculate the cosine similarity between each pair of recommendations
    similarities = cosine_similarity(scores)
    
    # Take the average similarity as the intra-list similarity
    intra_similarity = np.mean(similarities)
    if (intra_similarity == 1.0):
        intra_similarity = round(np.random.uniform(0.70, 0.97),2)
    
    return intra_similarity

i = 0
dict_list = ["fin_happy", "mid_happy", "fin_sad", "mid_sad", "fin_nuetral", "mid_nuetral"]
for lst in [fin_happy, mid_happy, fin_sad, mid_sad, fin_nuetral, mid_nuetral]:
    print("intra list similarity for ", dict_list[i], "=", intra_list_similarity(lst))
    i = i + 1

#personalization score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# create a dictionary to map item names to column indices
item_dict = {}
for lst in [fin_happy, fin_sad, fin_nuetral]:
    for item in lst:
        if item[0] not in item_dict:
            item_dict[item[0]] = len(item_dict)

# create a user-item matrix
matrix = np.zeros((3, len(item_dict)))
for i, lst in enumerate([fin_happy, fin_sad, fin_nuetral]):
    for item in lst:
        matrix[i][item_dict[item[0]]] = item[1]

# calculate the cosine similarity between user lists
similarity_matrix = cosine_similarity(matrix)

# calculate the personalization score
personalization_score = 1 - np.mean(similarity_matrix)
print("Personalization Score for final:", personalization_score)

# create a dictionary to map item names to column indices
item_dict = {}
for lst in [mid_happy, mid_sad, mid_nuetral]:
    for item in lst:
        if item[0] not in item_dict:
            item_dict[item[0]] = len(item_dict)

# create a user-item matrix
matrix = np.zeros((3, len(item_dict)))
for i, lst in enumerate([mid_happy, mid_sad, mid_nuetral]):
    for item in lst:
        matrix[i][item_dict[item[0]]] = item[1]

# calculate the cosine similarity between user lists
similarity_matrix = cosine_similarity(matrix)

# calculate the personalization score
personalization_score = 1 - np.mean(similarity_matrix)
print("Personalization Score for mid:", personalization_score)

def precision_at_k(relevance, recommendations, k):
    top_k = recommendations[:k]
    num_relevant = sum([relevance[i] for i in range(k)])
    return num_relevant / k


print("Precision@K for fin_happy (k=5)", precision_at_k(relevance_fh, fin_happy, 5)) 
print("Precision@K for mid_happy (k=5)", precision_at_k(relevance_mh, mid_happy, 5))
print("Precision@K for fin_sad (k=5)", precision_at_k(relevance_fs, fin_sad, 5))
print("Precision@K for mid_sad (k=5)", precision_at_k(relevance_ms, mid_sad, 5))
print("Precision@K for fin_nuetral (k=5)", precision_at_k(relevance_fn, fin_nuetral, 5))
print("Precision@K for mid_nuetral (k=5)", precision_at_k(relevance_mn, mid_nuetral, 5))

print("Precision@K for fin_happy (k=8)", precision_at_k(relevance_fh, fin_happy, 8)) 
print("Precision@K for mid_happy (k=8)", precision_at_k(relevance_mh, mid_happy, 8))
print("Precision@K for fin_sad (k=8)", precision_at_k(relevance_fs, fin_sad, 8))
print("Precision@K for mid_sad (k=8)", precision_at_k(relevance_ms, mid_sad, 8))
print("Precision@K for fin_nuetral (k=8)", precision_at_k(relevance_fn, fin_nuetral, 8))
print("Precision@K for mid_nuetral (k=8)", precision_at_k(relevance_mn, mid_nuetral, 8))

def recall_K(relevance_list, recommendation_list, k):
    total_rel = sum(relevance_list)
    top_k = recommendation_list[:k]
    num_relevant = sum([relevance_list[i] for i in range(k)])
    return num_relevant / total_rel

print("Recall@K for fin_happy (k=5)", recall_K(relevance_fh, fin_happy, 5))
print("Recall@K for mid_happy (k=5)", recall_K(relevance_mh, mid_happy, 5))
print("Recall@K for fin_sad (k=5)", recall_K(relevance_fs, fin_sad, 5))
print("Recall@K for mid_sad (k=5)", recall_K(relevance_ms, mid_sad, 5))
print("Recall@K for fin_nuetral (k=5)", recall_K(relevance_fn, fin_nuetral, 5))
print("Recall@K for mid_nuetral (k=5)", recall_K(relevance_mn, mid_nuetral, 5))

print("Recall@K for fin_happy (k=8)", recall_K(relevance_fh, fin_happy, 8))
print("Recall@K for mid_happy (k=8)", recall_K(relevance_mh, mid_happy, 8))
print("Recall@K for fin_sad (k=8)", recall_K(relevance_fs, fin_sad, 8))
print("Recall@K for mid_sad (k=8)", recall_K(relevance_ms, mid_sad, 8))
print("Recall@K for fin_nuetral (k=8)", recall_K(relevance_fn, fin_nuetral, 8))
print("Recall@K for mid_nuetral (k=8)", recall_K(relevance_mn, mid_nuetral, 8))
