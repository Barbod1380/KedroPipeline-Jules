import pandas as pd
import re



def load_rules(excel_file_path):
    """
    Preprocesses Excel file into matching rules
    Returns list of tuples: (conditions, number)
        conditions: list of (operator, phrase) tuples
        number: associated 3-digit code
    """
    try:
        df = pd.read_excel(excel_file_path, header=None, names=['pattern', 'number'])
        def remove_dots_regex(value):
            if pd.isna(value):
                return value
            return re.sub(r'\.', '', str(value))
        
        df = df.map(remove_dots_regex)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []
    
    rules = []
    for _, row in df.iterrows():
        pattern_str = str(row['pattern']).strip()
        number = int(row['number'])
        
        tokens = re.split('([+-])', pattern_str)
        tokens = [' '+token.strip()+' ' for token in tokens if token.strip()]
        if not tokens:
            continue
        conditions = []
        conditions.append((' + ', tokens[0]))
        
        i = 1
        while i < len(tokens):
            operator = tokens[i]
            phrase = tokens[i+1] if i+1 < len(tokens) else ''
            if operator in (' + ', ' - ') and phrase:
                conditions.append((operator, phrase))
            i += 2
        rules.append((conditions, number))
    return rules,df



def process_phrases_to_dict_to_handle_underlines(df):
    """
    Processes an Excel file and returns a replacement dictionary directly
    Args:
        input_file: Path to input Excel file (.xlsx)
    Returns:
        Dictionary mapping {phrase_with_spaces: phrase_with_underscores}
    """
    
    # Dictionary to store replacements
    replacement_dict = {}
    
    # Process each cell in the DataFrame
    for column in df.columns:
        for cell in df[column]:
            if pd.notna(cell) and isinstance(cell, str):
                # Split by + or - and strip whitespace
                split_phrases = re.split(r'[+-]', cell)
                for phrase in split_phrases:
                    phrase = phrase.strip()
                    if phrase and '_' in phrase:  # Only keep phrases with underscores
                        space_version = phrase.replace('_', ' ')
                        replacement_dict[space_version] = phrase
    
    print(f"Processed {len(replacement_dict)} phrase replacements")
    return replacement_dict



def replace_phrases(input_string, replacement_dict):
    """Replace phrases in input string according to replacement dictionary"""
    # Sort replacements by length (longest first) to handle multi-word phrases correctly
    sorted_replacements = sorted(replacement_dict.items(), 
                               key=lambda x: len(x[0]), 
                               reverse=True)
    
    for space_phrase, underscore_phrase in sorted_replacements:
        input_string = input_string.replace(space_phrase, underscore_phrase)
    return input_string