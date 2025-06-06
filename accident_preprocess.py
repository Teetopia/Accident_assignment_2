import pandas as pd
import matplotlib.pyplot as plt

# Weather severity weight base on Domain knowledge
weather_level = {  
    "Clear" : 1,
    "Strong winds" : 2,
    "Raining" : 3,
    "Dust" : 4,
    "Smoke" : 5,
    "Fog" : 6,
    "Snowing" : 7,
}

# Road surface severity weight base on Domain knowledge
road_level = {  
    "Dry" : 1,
    "Wet" : 2,
    "Muddy" : 3,
    "Snowy" : 4,
    "Icy" : 5,
}

# Find weighted average
def get_avg_index(desc, level):
    # Split description(s) in to list
    conditions = [c.strip() for c in desc.split(',')]

    values = [level[c] for c in conditions]
    
    # Use index-based weights (1-based)
    weights = list(range(1, len(values) + 1))
    
    # Calculate weighted average
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    total_weight = sum(weights)

    return round(weighted_sum / total_weight, 2)

# get sevirt index value based on number of people involved 
def get_severity_index(row):
    if row['NO_PERSONS_KILLED'] >= 1:
        return 400 + row['NO_PERSONS_KILLED']
    elif row['NO_PERSONS_INJ_2'] >= 1:
        return 300 + row['NO_PERSONS_INJ_2'] 
    elif row['NO_PERSONS_INJ_3'] >= 1:
        return 200 + row['NO_PERSONS_INJ_3']
    elif row['NO_PERSONS_NOT_INJ'] >= 1:
        return 100
    else:
        return 0

# def get_severity_index(row):
#     killed_weight = 50
#     serious_injury_weight = 5
#     minor_injury_weight = 1
#     return  row['NO_PERSONS_KILLED'] * killed_weight  + row['NO_PERSONS_INJ_2'] * serious_injury_weight  +  row['NO_PERSONS_INJ_3'] * minor_injury_weight

# Flip SEVERITY value and remove consistency
def handle_severity(row):
    if row['NO_PERSONS_KILLED'] >= 1:
        return 4
    elif row['NO_PERSONS_INJ_2'] >= 1:
        return 3
    elif row['NO_PERSONS_INJ_3'] >= 1:
        return 2
    elif row['NO_PERSONS_NOT_INJ'] >= 1:
        return 1
    else:
        return 0 


# Merge atmosphere condition so that each entry is unique
def merge_atmosphere():
    atmosphere = pd.read_csv('atmospheric_cond.csv')

    # Ignore Not Known value
    atmosphere = atmosphere[atmosphere['ATMOSPH_COND_DESC'] != 'Not known']

    atmosphere_sorted = atmosphere.sort_values(by=['ACCIDENT_NO', 'ATMOSPH_COND_SEQ'])

    # Group description that have more than 1 into single unique ACCIDENT_NO
    merged = atmosphere_sorted.groupby('ACCIDENT_NO').agg({
        'ATMOSPH_COND_DESC': lambda x: ', '.join(x)
    }).reset_index()

    merged['ATMOSPH_INDEX'] = [get_avg_index(i, weather_level) for i in merged['ATMOSPH_COND_DESC']]
    merged.to_csv("merged_atmospheric.csv", index=False)

    return

# Merge road surface condition so that each entry is unique
def merge_road():
    road_surface = pd.read_csv('road_surface_cond.csv')

    # Ignore Not Known value
    road_surface = road_surface[road_surface['SURFACE_COND_DESC'] != 'Unk.']

    road_surface_sorted = road_surface.sort_values(by=['ACCIDENT_NO', 'SURFACE_COND_SEQ'])

    # Group description that have more than 1 into single unique ACCIDENT_NO
    merged = road_surface_sorted.groupby('ACCIDENT_NO').agg({
        'SURFACE_COND_DESC': lambda x: ', '.join(x)
    }).reset_index()

    merged['SURFACE_INDEX'] = [get_avg_index(i, road_level) for i in merged['SURFACE_COND_DESC']]
    merged.to_csv("merged_road_surface.csv", index=False)

    return

# Merge atmosphere and road surface into accident dataset
def merge_accident():

    merge_atmosphere()
    merge_road()
    accident = pd.read_csv('accident.csv')
    merged_atmosphere = pd.read_csv('merged_atmospheric.csv')
    merged_road_surface = pd.read_csv('merged_road_surface.csv')
    
    merged_accident = accident.merge(merged_atmosphere[['ACCIDENT_NO', 'ATMOSPH_INDEX', 'ATMOSPH_COND_DESC']], on='ACCIDENT_NO', how='left')
    merged_accident = merged_accident.merge(merged_road_surface[['ACCIDENT_NO', 'SURFACE_INDEX', 'SURFACE_COND_DESC']], on='ACCIDENT_NO', how='left')
    
    merged_accident = merged_accident[
        merged_accident['ATMOSPH_COND_DESC'].notna() & 
        merged_accident['SURFACE_COND_DESC'].notna()
    ]

    merged_accident['SEVERITY'] = merged_accident.apply(handle_severity, axis=1)
    merged_accident['SEVERITY_INDEX'] = merged_accident.apply(get_severity_index, axis=1)

    merged_accident.to_csv("merged_accident.csv", index=False)
    return

merge_accident()