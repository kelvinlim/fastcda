import yaml

def read_yaml(file_path):
    """
    Reads a YAML file and returns its content as a dictionary.

    :param file_path: Path to the YAML file.
    :return: Dictionary containing the YAML file content.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

if __name__ == "__main__":


    file_path = "fastcda.yaml"
    try:
        content = read_yaml(file_path)
        print(content)
    except Exception as e:
        print(f"Error reading YAML file: {e}")
