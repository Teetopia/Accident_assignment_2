import pandas as pd
import matplotlib.pyplot as plt

weather_level = {  
    "Clear" : 1,
    "Strong winds" : 2,
    "Raining" : 3,
    "Dust" : 4,
    "Smoke" : 5,
    "Fog" : 6,
    "Snowing" : 7,
}

road_level = {  
    "Dry" : 1,
    "Wet" : 2,
    "Muddy" : 3,
    "Snowy" : 4,
    "Icy" : 5,
}

def get_avg_index(desc, level):
    conditions = [c.strip() for c in desc.split(',')]
    indices = [level.get(cond, None) for cond in conditions]
    indices = [i for i in indices if i is not None] 
    return round(sum(indices) / len(indices), 2) if indices else None

def merge_atmosphere():
    atmosphere = pd.read_csv('atmospheric_cond.csv')
    atmosphere = atmosphere[atmosphere['ATMOSPH_COND_DESC'] != 'Not known']

    atmosphere_sorted = atmosphere.sort_values(by=['ACCIDENT_NO', 'ATMOSPH_COND_SEQ'])

    merged = atmosphere_sorted.groupby('ACCIDENT_NO').agg({
        'ATMOSPH_COND_DESC': lambda x: ', '.join(x)
    }).reset_index()

    merged['ATMOSPH_INDEX'] = [get_avg_index(i, weather_level) for i in merged['ATMOSPH_COND_DESC']]
    merged.to_csv("merged_atmospheric.csv", index=False)

    return

def merge_road():
    road_surface = pd.read_csv('road_surface_cond.csv')
    road_surface = road_surface[road_surface['SURFACE_COND_DESC'] != 'Unk.']

    road_surface_sorted = road_surface.sort_values(by=['ACCIDENT_NO', 'SURFACE_COND_SEQ'])

    merged = road_surface_sorted.groupby('ACCIDENT_NO').agg({
        'SURFACE_COND_DESC': lambda x: ', '.join(x)
    }).reset_index()

    merged['SURFACE_INDEX'] = [get_avg_index(i, road_level) for i in merged['SURFACE_COND_DESC']]
    merged.to_csv("merged_road_surface.csv", index=False)

    return
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

    merged_accident.to_csv("merged_accident.csv", index=False)
    return


merge_accident()