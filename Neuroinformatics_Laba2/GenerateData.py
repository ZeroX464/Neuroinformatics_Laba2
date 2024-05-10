import numpy as np
import pandas as pd

# ��������� ��������� �������
def generate_dataset(num_samples):
    # ������� ��� �������� �������� ���������� ��� ������� ������ ��������
    parameters = {
        "Meadow Grasses": {
            "average_height": (20, 100),
            "average_width": (5, 30),
            "growth_rate": (0.2, 0.8),
            "watering_needs": (1, 3),
            "sunlight_requirements": (4, 7),
            "soil_requirements": (2, 5),
            "blooming_season": (5, 8),
        },
        "Desert Plants": {
            "average_height": (10, 50),
            "average_width": (5, 20),
            "growth_rate": (0.1, 0.5),
            "watering_needs": (1, 2),
            "sunlight_requirements": (7, 10),
            "soil_requirements": (1, 3),
            "blooming_season": (4, 7),
        },
        "Aquatic Plants": {
            "average_height": (10, 200),
            "average_width": (5, 150),
            "growth_rate": (0.3, 1.5),
            "watering_needs": (1, 5),
            "sunlight_requirements": (1, 4),
            "soil_requirements": (1, 3),
            "blooming_season": (5, 9),
        },
        "Tropical Palms": {
            "average_height": (100, 500),
            "average_width": (50, 300),
            "growth_rate": (0.5, 2.0),
            "watering_needs": (3, 5),
            "sunlight_requirements": (8, 10),
            "soil_requirements": (3, 6),
            "blooming_season": (5, 10),
        },
        "Mountain Flowers": {
            "average_height": (10, 60),
            "average_width": (5, 30),
            "growth_rate": (0.1, 0.7),
            "watering_needs": (2, 4),
            "sunlight_requirements": (3, 8),
            "soil_requirements": (2, 5),
            "blooming_season": (6, 9),
        },
        "Succulents": {
            "average_height": (5, 50),
            "average_width": (5, 30),
            "growth_rate": (0.1, 0.5),
            "watering_needs": (1, 3),
            "sunlight_requirements": (7, 10),
            "soil_requirements": (4, 7),
            "blooming_season": (4, 8),
        },
        "Exotic Orchids": {
            "average_height": (10, 100),
            "average_width": (5, 50),
            "growth_rate": (0.3, 1.2),
            "watering_needs": (2, 4),
            "sunlight_requirements": (3, 7),
            "soil_requirements": (3, 5),
            "blooming_season": (6, 10),
        },
        "Alpine Shrubs": {
            "average_height": (30, 200),
            "average_width": (30, 150),
            "growth_rate": (0.2, 0.8),
            "watering_needs": (2, 4),
            "sunlight_requirements": (4, 8),
            "soil_requirements": (3, 6),
            "blooming_season": (5, 9),
        },
        "Fruit Trees": {
            "average_height": (100, 500),
            "average_width": (50, 300),
            "growth_rate": (0.5, 1.5),
            "watering_needs": (3, 5),
            "sunlight_requirements": (6, 9),
            "soil_requirements": (3, 6),
            "blooming_season": (4, 8),
        },
        "Flowering Shrubs": {
            "average_height": (50, 300),
            "average_width": (50, 250),
            "growth_rate": (0.3, 1.0),
            "watering_needs": (2, 4),
            "sunlight_requirements": (4, 8),
            "soil_requirements": (3, 6),
            "blooming_season": (4, 9),
        }
    }

    # ������ ��� �������� ������ �� ��������� �������
    data = []

    # ��������� ������ ��� ������� ������ ��������
    for plant_class, params in parameters.items():
        # ����������� ���������� ������������� ��� ������� ������
        if plant_class in ["Meadow Grasses", "Mountain Flowers", "Alpine Shrubs"]:
            origin_mainland = 1  # �������
        elif plant_class in ["Desert Plants", "Succulents"]:
            origin_mainland = 2  # ������
        elif plant_class in ["Aquatic Plants"]:
            origin_mainland = 3  # ����� �������
        elif plant_class in ["Tropical Palms"]:
            origin_mainland = 6  # ����
        elif plant_class in ["Exotic Orchids", "Fruit Trees"]:
            origin_mainland = 4  # ���-��������� ����
        elif plant_class in ["Flowering Shrubs"]:
            origin_mainland = 5  # �������� �������
        else:
            origin_mainland = -1  # ��������� ������������� ����������
        
        for _ in range(num_samples):
            sample = {
                "origin_mainland": origin_mainland
            }
            for param, (min_val, max_val) in params.items():
                sample[param] = np.random.uniform(min_val, max_val)
            sample["plant_class"] = plant_class  # ���������� ������� � ��������� ������ ��������
            data.append(sample)

    # �������� DataFrame �� ��������������� ������
    df = pd.DataFrame(data)
    return df

# ��������� ��������� ������� �������� 300
dataset = generate_dataset(300)

# ���������� � ������� Excel (xlsx)
with pd.ExcelWriter("plant_dataset.xlsx") as writer:
    dataset.to_excel(writer, index=False)
