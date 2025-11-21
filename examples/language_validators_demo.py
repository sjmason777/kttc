"""
Language Validators Integration Examples.

This script demonstrates how to use language-specific validators
for Russian, Chinese, Hindi, and Persian translation quality assessment.
"""

from kttc.terminology import (
    ChineseMeasureWordValidator,
    HindiPostpositionValidator,
    LanguageValidatorFactory,
    PersianEzafeValidator,
    RussianCaseAspectValidator,
)


def russian_validator_example():
    """Demonstrate Russian case and aspect validation."""
    print("\n" + "=" * 70)
    print("RUSSIAN CASE AND ASPECT VALIDATOR")
    print("=" * 70)

    validator = RussianCaseAspectValidator()

    # Example 1: Get information about Russian cases
    print("\n1. Case Information:")
    print("-" * 50)
    for case in ["nominative", "genitive", "dative", "accusative"]:
        case_info = validator.get_case_info(case)
        if case_info:
            print(f"\n{case.capitalize()} case ({case_info['name_ru']}):")
            print(f"  Question: {case_info['question']}")
            print(f"  Function: {case_info['function']}")
            print(f"  Example: {case_info['example']}")

    # Example 2: Get aspect information
    print("\n2. Aspect Information:")
    print("-" * 50)
    for aspect in ["perfective", "imperfective"]:
        aspect_info = validator.get_aspect_info(aspect)
        if aspect_info:
            print(f"\n{aspect.capitalize()} aspect ({aspect_info['name_ru']}):")
            print("  Characteristics:")
            for char in aspect_info["when_to_use"]:
                print(f"    - {char}")
            print(f"  Examples: {aspect_info['examples']}")

    # Example 3: Validate preposition-case agreement
    print("\n3. Preposition-Case Validation:")
    print("-" * 50)
    test_cases = [
        ("в", "prepositional"),
        ("из", "genitive"),
        ("к", "dative"),
    ]
    for prep, case in test_cases:
        is_valid, error = validator.validate_case_preposition(prep, case)
        status = "✓ Valid" if is_valid else f"✗ Invalid: {error}"
        print(f"  Preposition '{prep}' + {case} case: {status}")

    # Example 4: Get aspect usage rules
    print("\n4. Aspect Usage Rules:")
    print("-" * 50)
    rules = validator.get_aspect_usage_rules("perfective")
    print("  Perfective aspect should be used for:")
    for rule in rules["when_to_use"]:
        print(f"    - {rule}")


def chinese_validator_example():
    """Demonstrate Chinese measure word (classifier) validation."""
    print("\n" + "=" * 70)
    print("CHINESE MEASURE WORD VALIDATOR")
    print("=" * 70)

    validator = ChineseMeasureWordValidator()

    # Example 1: Get classifiers by category
    print("\n1. Classifiers by Category:")
    print("-" * 50)
    individual = validator.get_classifier_by_category("individual_classifiers")
    print(f"  Individual classifiers found: {len(individual)}")
    for clf_char, clf_info in list(individual.items())[:3]:
        print(f"\n  {clf_char} ({clf_info.get('pinyin', 'N/A')}):")
        print(f"    Usage: {clf_info.get('usage', 'N/A')}")
        examples = clf_info.get("examples", [])
        if examples:
            print(f"    Examples: {examples[0]}")

    # Example 2: Get most common classifiers
    print("\n2. Most Common Classifiers:")
    print("-" * 50)
    common = validator.get_most_common_classifiers(limit=5)
    for i, (clf_char, clf_info) in enumerate(common, 1):
        pinyin = clf_info.get("pinyin", "N/A")
        usage = clf_info.get("usage", "N/A")
        print(f"  {i}. {clf_char} ({pinyin}): {usage}")

    # Example 3: Find classifier for specific nouns
    print("\n3. Finding Classifiers for Nouns:")
    print("-" * 50)
    test_nouns = ["人", "本", "只"]
    for noun in test_nouns:
        result = validator.get_classifier_for_noun(noun)
        if result:
            print(f"\n  Noun '{noun}':")
            print(f"    Classifier: {result['classifier']} ({result['pinyin']})")
            print(f"    Category: {result['category']}")
            print(f"    Usage: {result['usage']}")
        else:
            print(f"  Noun '{noun}': No specific classifier found")

    # Example 4: Validate classifier-noun pairs
    print("\n4. Classifier-Noun Pair Validation:")
    print("-" * 50)
    test_pairs = [
        ("个", "人"),
        ("本", "书"),
        ("条", "狗"),
    ]
    for clf, noun in test_pairs:
        is_valid, message = validator.validate_classifier_noun_pair(clf, noun)
        status = "✓ Valid" if is_valid else f"✗ {message}"
        print(f"  {clf} + {noun}: {status}")


def hindi_validator_example():
    """Demonstrate Hindi postposition and case validation."""
    print("\n" + "=" * 70)
    print("HINDI POSTPOSITION VALIDATOR")
    print("=" * 70)

    validator = HindiPostpositionValidator()

    # Example 1: Get information about Hindi cases
    print("\n1. Hindi Case System (8 cases):")
    print("-" * 50)
    case_names = [
        "Nominative (कर्ता)",
        "Accusative (कर्म)",
        "Instrumental (करण)",
        "Dative (संप्रदान)",
        "Ablative (अपादान)",
        "Genitive (संबंध)",
        "Locative (अधिकरण)",
        "Vocative (संबोधन)",
    ]
    for case_num in range(1, 9):
        case_info = validator.get_case_info(case_num)
        if case_info:
            print(f"\n  Case {case_num}: {case_names[case_num - 1]}")
            postposition = case_info.get("postposition", "N/A")
            print(f"    Postposition: {postposition}")
            examples = case_info.get("examples", {})
            if isinstance(examples, dict):
                # Examples is a dict with keys like 'without_ne', 'with_ne', etc.
                for key, ex_list in examples.items():
                    if ex_list and len(ex_list) > 0:
                        print(f"    Example ({key}): {ex_list[0]}")
                        break
            elif isinstance(examples, list) and len(examples) > 0:
                # Examples is a simple list
                print(f"    Example: {examples[0]}")

    # Example 2: Validate ergative construction
    print("\n2. Ergative Construction Validation:")
    print("-" * 50)
    test_cases = [
        ("transitive", "simple_past", True, "Valid: transitive + past + ने"),
        ("transitive", "simple_past", False, "Invalid: missing ने"),
        ("intransitive", "simple_past", False, "Valid: intransitive, no ने"),
        ("intransitive", "simple_past", True, "Invalid: intransitive with ने"),
    ]
    for verb_type, tense, has_ne, description in test_cases:
        is_valid, error = validator.validate_ergative_construction(verb_type, tense, has_ne)
        status = "✓" if is_valid else "✗"
        result = error if error else "Correct"
        print(f"  {status} {description}")
        if error:
            print(f"      → {result}")

    # Example 3: Get postpositions for cases
    print("\n3. Postpositions for Each Case:")
    print("-" * 50)
    for case_num in [2, 3, 6, 7]:  # Sample cases
        postposition = validator.get_postposition_for_case(case_num)
        if postposition:
            print(f"  Case {case_num}: {postposition}")

    # Example 4: Oblique form rules
    print("\n4. Oblique Form Rules:")
    print("-" * 50)
    oblique_rules = validator.get_oblique_form_rule()
    if oblique_rules:
        print(f"  Definition: {oblique_rules.get('definition', 'N/A')}")
        when_used = oblique_rules.get("when_used", [])
        if when_used:
            print("  Used when:")
            for rule in when_used[:3]:
                print(f"    - {rule}")


def persian_validator_example():
    """Demonstrate Persian ezafe and grammar validation."""
    print("\n" + "=" * 70)
    print("PERSIAN EZAFE VALIDATOR")
    print("=" * 70)

    validator = PersianEzafeValidator()

    # Example 1: Get ezafe construction rules
    print("\n1. Ezafe Construction Rules:")
    print("-" * 50)
    ezafe_rules = validator.get_ezafe_rules()
    if ezafe_rules:
        print(f"  Definition: {ezafe_rules.get('definition', 'N/A')}")
        print(f"  Pronunciation: {ezafe_rules.get('pronunciation', 'N/A')}")
        functions = ezafe_rules.get("functions", [])
        if functions:
            print("  Functions:")
            for func in functions[:3]:
                print(f"    - {func}")

    # Example 2: Validate ezafe usage
    print("\n2. Ezafe Usage Validation:")
    print("-" * 50)
    construction_types = ["noun_adjective", "noun_noun", "noun_prepositional_phrase"]
    for const_type in construction_types:
        is_valid, info = validator.validate_ezafe_usage(const_type)
        if is_valid:
            print(f"\n  ✓ {const_type}:")
            examples = info.get("examples", [])
            if examples and len(examples) > 0:
                print(f"      Example: {examples[0]}")
        else:
            print(f"  ✗ {const_type}: {info.get('error', 'Unknown')}")

    # Example 3: Compound verb information
    print("\n3. Compound Verb Information:")
    print("-" * 50)
    light_verbs = ["کردن", "شدن", "زدن", "دادن"]
    for lv in light_verbs:
        info = validator.get_compound_verb_info(lv)
        if info:
            print(f"\n  Light verb: {lv}")
            examples = info.get("examples", [])
            if isinstance(examples, list) and len(examples) > 0:
                print(f"    Example: {examples[0]}")
            elif isinstance(examples, dict):
                # Examples might be a dict with keys
                for key, ex_val in examples.items():
                    if isinstance(ex_val, list) and len(ex_val) > 0:
                        print(f"    Example: {ex_val[0]}")
                        break
                    elif isinstance(ex_val, str):
                        print(f"    Example: {ex_val}")
                        break

    # Example 4: Validate object marker 'را' usage
    print("\n4. Object Marker 'را' Validation:")
    print("-" * 50)
    test_cases = [
        ("definite", True, "Definite object with را"),
        ("definite", False, "Definite object without را"),
        ("indefinite", False, "Indefinite object without را"),
        ("indefinite", True, "Indefinite object with را"),
        ("proper_noun", True, "Proper noun with را"),
        ("pronoun", True, "Pronoun with را"),
    ]
    for obj_type, has_ra, description in test_cases:
        is_valid, error = validator.validate_object_marker_ra(obj_type, has_ra)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"  {status}: {description}")
        if error:
            print(f"      → {error}")


def factory_pattern_example():
    """Demonstrate using the factory pattern to create validators."""
    print("\n" + "=" * 70)
    print("LANGUAGE VALIDATOR FACTORY")
    print("=" * 70)

    print("\n1. Available Validators:")
    print("-" * 50)
    available = LanguageValidatorFactory.list_available_validators()
    language_names = {
        "ru": "Russian (Русский)",
        "zh": "Chinese (中文)",
        "hi": "Hindi (हिन्दी)",
        "fa": "Persian (فارسی)",
    }
    for lang_code in available:
        print(f"  - {lang_code}: {language_names.get(lang_code, 'Unknown')}")

    print("\n2. Creating Validators via Factory:")
    print("-" * 50)
    for lang_code in available:
        validator = LanguageValidatorFactory.create_validator(lang_code)
        validator_class = validator.__class__.__name__
        print(f"  {lang_code} → {validator_class}")

    print("\n3. Error Handling for Unsupported Languages:")
    print("-" * 50)
    try:
        validator = LanguageValidatorFactory.create_validator("xx")
        print("  Unexpected: validator created for unsupported language")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")


def main():
    """Run all validator examples."""
    print("\n" + "=" * 70)
    print(" KTTC LANGUAGE VALIDATORS - INTEGRATION EXAMPLES")
    print("=" * 70)
    print("\nThis demo showcases language-specific grammatical validators")
    print("for translation quality assessment in 4 languages.")

    # Run all examples
    russian_validator_example()
    chinese_validator_example()
    hindi_validator_example()
    persian_validator_example()
    factory_pattern_example()

    print("\n" + "=" * 70)
    print(" DEMO COMPLETE")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - glossaries/README.md")
    print("  - src/kttc/terminology/language_validators.py")
    print("  - tests/unit/test_language_validators.py")
    print()


if __name__ == "__main__":
    main()
