import random
import json

def generate_synthetic_therapy(num_records):
    templates_yes = [
        "Касае се за {gender} на {age} г. и половина, при {observed_condition}. В рамките на няколко месеца е имала {number_of_issues} {issues} с необходимост от {treatment}. Моделът на боледуване е основно {symptom} с {additional_symptom}. Множеството инфекции в последните няколко месеца са свързани със {cause}. Назначените имунологични изследвания показват {immunological_findings}. Детето е приемало {supplements}, но досега не е провеждана {therapy}.",
        # Add more templates indicating immune deficiency as needed
    ]
    
    templates_no = [
        "Касае се за {gender} на {age} г., което се радва на добро здраве. През последните няколко месеца не е имало значителни заболявания. Назначените имунологични изследвания показват {immunological_findings}. Детето е приемало {supplements}.",
        # Add more templates indicating no immune deficiency as needed
    ]

    genders = ["момиче", "момче"]
    conditions = [
        "наблюдава повишен брой боледувания, които се усложняват",
        "чести инфекции на горните дихателни пътища",
        "постоянна кашлица и хрема"
    ]
    issues = ["пневмонии", "синузити", "бронхити"]
    treatments = ["хоспитализация и приложение на антибиотици", "амбулаторно лечение", "домашно лечение"]
    symptoms = ["задна хрема", "остра кашлица", "температура"]
    additional_symptoms = ["кашлица", "хрипове", "отпадналост"]
    causes = ["посещение на детско заведение", "контакт с болни деца", "сезонни промени"]
    immunological_findings_yes = [
        "намалени нива на хуморалния имунитет",
        "намалени нива на клетъчния имунитет"
    ]
    immunological_findings_no = [
        "нормални нива на хуморалния имунитет",
        "нормални нива на клетъчния имунитет"
    ]
    supplements = [
        "почти всички видове добавки-имуномодулатори",
        "витамини и минерали",
        "пробиотици и имуностимуланти"
    ]
    therapies = ["насочена имунопрофилактика", "специфична ваксинация", "имунотерапия"]

    synthetic_therapies = []
    for _ in range(num_records):
        if random.random() > 0.5:
            template = random.choice(templates_yes)
            synthetic_therapy = template.format(
                gender=random.choice(genders),
                age=random.randint(1, 12),
                observed_condition=random.choice(conditions),
                number_of_issues=random.randint(1, 5),
                issues=random.choice(issues),
                treatment=random.choice(treatments),
                symptom=random.choice(symptoms),
                additional_symptom=random.choice(additional_symptoms),
                cause=random.choice(causes),
                immunological_findings=random.choice(immunological_findings_yes),
                supplements=random.choice(supplements),
                therapy=random.choice(therapies)
            )
            label = 'YES'
        else:
            template = random.choice(templates_no)
            synthetic_therapy = template.format(
                gender=random.choice(genders),
                age=random.randint(1, 12),
                immunological_findings=random.choice(immunological_findings_no),
                supplements=random.choice(supplements)
            )
            label = 'NO'
        synthetic_therapies.append((synthetic_therapy, label))
    
    return synthetic_therapies

if __name__ == "__main__":
    print("Script started...")
    num_records = 100
    print("Generating synthetic therapies...")
    synthetic_therapies = generate_synthetic_therapy(num_records)
    print("Finished generating synthetic therapies.")
    output_path = '../data/synthetic_therapies.json'
    print(f"Ensuring directory exists: {output_path}")
    with open(output_path, 'w') as f:
        print(f"Saving synthetic therapies to {output_path}...")
        json.dump(synthetic_therapies, f, ensure_ascii=False, indent=4)
    print(f"Generated {num_records} synthetic therapy records and saved to {output_path}")
    print("Script finished.")
