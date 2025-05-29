import json

features_persona = [
    ("Young", "Older"),
    ("Female", "Male"),
    ("High Neuroticism", "Low Neuroticism"),
    ("High Extraversion", "Low Extraversion"),
    ("High Openness", "Low Openness"),
    ("High Agreeableness", "Low Agreeableness"),
    ("High Conscientiousness", "Low Conscientiousness"),
    ("Likes a certain food", "Dislikes a certain food"),
    ("Likes a certain living environment", "Dislikes a certain living environment"),
    ("Likes sleep", "Dislikes sleep"),
    ("Aggressive investment", "Conservative investment"),
    ("Good at saving", "Bad at saving"),
    ("Concerned about physical safety", "Not concerned about physical safety"),
    ("Concerned about environmental safety", "Not concerned about environmental safety"),
    ("Prefers superficial interaction (casual, stress-free chat)", "Prefers deep interaction (discussing interests, emotional topics, etc.)"),
    ("Prefers direct communication to handle conflict", "Prefers avoidance, mediation, compromise to handle conflict"),
    ("Concise communication style", "Detailed communication style"),
    ("Strong need for a certain work environment", "Indifferent to work environment needs"),
    ("Strong need for recognition from others", "Indifferent to recognition from others"),
    ("Strong need for personal achievement", "Indifferent to personal achievement"),
    ("Likes a certain area of knowledge", "Dislikes a certain area of knowledge"),
    ("Likes a certain learning style", "Dislikes a certain learning style"),
    ("Likes a certain form of creative expression (e.g., art, writing, music)", "Dislikes a certain form of creative expression (e.g., art, writing, music)"),
    ("Strong need for Order (neatness, organization, avoiding chaos)", "Indifferent to orderliness"),
    ("Strong need for Retention (holding onto objects, unwilling to lose or change)", "Indifferent to retention (unconcerned about keeping objects)"),
    ("Strong need for Inviolacy (maintaining dignity and reputation)", "Indifferent to inviolacy (unconcerned with dignity or reputation)"),
    ("Strong need for Infavoidance (avoiding failure and embarrassment)", "Indifferent to Infavoidance (unconcerned with failure or embarrassment)"),
    ("Strong need for Counteraction (overcoming failure and obstacles)", "Indifferent to Counteraction (unconcerned with failure)"),
    ("Strong need for Seclusion (desire for isolation from others)", "Indifferent to Seclusion (does not care about isolation)"),
    ("Strong need for Dominance (controlling others through command or persuasion)", "Indifferent to Dominance (does not care about control)"),
    ("Strong need for Deference (following authority or rules)", "Indifferent to Deference (does not care about authority)"),
    ("Strong need for Autonomy (pursuing independence and self-reliance)", "Indifferent to Autonomy (does not care about independence)"),
    ("Strong need for Contrariance (pursuing uniqueness, opposing the norm)", "Indifferent to Contrariance (does not seek uniqueness)"),
    ("Strong need for Abasement (accepting blame, enjoying pain or misfortune)", "Indifferent to Abasement (does not accept blame or enjoy misfortune)"),
    ("Strong need for Aggression (controlling others through forceful means)", "Indifferent to Aggression (does not engage in aggression)"),
    ("Strong need for Affiliation (desiring close relationships)", "Indifferent to Affiliation (does not care about close relationships)"),
    ("Strong need for Rejection (isolating oneself from negatively evaluated people)", "Indifferent to Rejection (does not care about social exclusion)"),
    ("Strong need for Nurturance (caring for others, protecting them from danger)", "Indifferent to Nurturance (does not care about nurturing others)"),
    ("Strong need for Succorance (desiring help, love, and comfort from others)", "Indifferent to Succorance (does not rely on others for comfort)"),
    ("Strong need for Play (enjoying fun, relaxation, and laughter)", "Indifferent to Play (does not prioritize fun or relaxation)"),
    ("Concerned about harmlessness", "Indifferent about harmlessness"),
    ("Concerned about instruction-following", "Indifferent about instruction-following"),
    ("Concerned about honesty", "Indifferent about honesty"),
    ("Concerned about truthfulness", "Indifferent about truthfulness"),
    ("Concerned about helpfulness", "Indifferent about helpfulness"),
    ("Concerned about coherence", "Indifferent about coherence"),
    ("Concerned about complexity", "Indifferent about complexity"),
    ("Likes science", "Dislikes science"),
    ("Likes knowledge", "Dislikes knowledge"),
    ("Likes psychology", "Dislikes psychology"),
    ("Likes cinema", "Dislikes cinema"),
    ("Likes entertainment", "Dislikes entertainment"),
    ("Likes gaming", "Dislikes gaming"),
    ("Likes parenting", "Dislikes parenting"),
    ("Likes wild imagination", "Dislikes wild imagination"),
    ("Likes anime", "Dislikes anime"),
    ("Likes sports", "Dislikes sports"),
    ("Likes law", "Dislikes law"),
    ("Likes workplace", "Dislikes workplace"),
    ("Likes pets", "Dislikes pets"),
    ("Likes travel", "Dislikes travel"),
    ("Likes health", "Dislikes health"),
    ("Likes stories", "Dislikes stories"),
    ("Likes cars", "Dislikes cars"),
    ("Likes gourmet food", "Dislikes gourmet food"),
    ("Likes education", "Dislikes education"),
    ("Likes current events", "Dislikes current events"),
    ("Likes home decor", "Dislikes home decor"),
    ("Likes international", "Dislikes international"),
    ("Likes finance", "Dislikes finance"),
    ("Likes campus life", "Dislikes campus life"),
    ("Likes digital technology", "Dislikes digital technology"),
    ("Likes emotions", "Dislikes emotions"),
    ("Likes humor", "Dislikes humor"),
    ("Likes music", "Dislikes music"),
    ("Likes reading", "Dislikes reading"),
    ("Likes painting", "Dislikes painting"),
    ("Likes dance", "Dislikes dance"),
    ("Likes crafts", "Dislikes crafts"),
    ("Likes photography", "Dislikes photography"),
    ("Likes culture", "Dislikes culture"),
    ("Likes fitness", "Dislikes fitness"),
    ("Likes art", "Dislikes art"),
    ("Likes stationery and planners", "Dislikes stationery and planners"),
    ("Likes celebrities", "Dislikes celebrities"),
    ("Likes outdoors", "Dislikes outdoors"),
    ("Likes camping", "Dislikes camping"),
    ("Likes social sciences", "Dislikes social sciences"),
    ("Likes weddings", "Dislikes weddings"),
    ("Likes fashion", "Dislikes fashion")
]

def trans_persona(embedding):
    description = []

    for i, value in enumerate(embedding):
        if value == 1:
            description.append(features_persona[i][0])  
        elif value == 0:
            description.append(features_persona[i][1])

    result = ', '.join(description)
    return result

def get_profile(history):
    embedding = []
    for it in history:
        features = it["Preference Direction"]
        embedding.append(features)
    mean_embedding = torch.tensor(embedding).mean(dim=0, keepdim=True)
    mean_embedding_list = mean_embedding.squeeze().tolist()
    # print(mean_embedding_list)
    # breakpoint()
    binary_emb = [
        1 if x >= 0.6 else 0 if x <= 0.4 else 0.5 
        for x in mean_embedding_list
    ]

    profile = trans_persona(binary_emb)

    if profile == "":
        ########
        mean_embedding = torch.tensor(embedding).mean(dim=0, keepdim=True)
        mean_embedding = mean_embedding.squeeze()

        max_val = torch.max(mean_embedding)
        min_val = torch.min(mean_embedding)
        if 1 - max_val < min_val:
            mean_embedding[mean_embedding == max_val] = 1
        else:
            mean_embedding[mean_embedding == min_val] = 0
        
        mean_embedding = torch.where((mean_embedding != 1) & (mean_embedding != 0), torch.tensor(0.5), mean_embedding)

        binary_emb = mean_embedding.tolist()
        profile = trans_persona(binary_emb)
        ########

    return profile