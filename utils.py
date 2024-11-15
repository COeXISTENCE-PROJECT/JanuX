import json
from prettytable import PrettyTable
import xml.etree.ElementTree as ET

def df_to_prettytable(df, header_message=None, print_every=1):
    table = PrettyTable()
    table.field_names = df.columns.tolist()
    for index, row in df.iterrows():
        if not (index % print_every):
            table.add_row(row.tolist())
    if header_message:
        print(f"##### {header_message} #####")
    print(table)


def get_params(file_path):      # Read params.json, resolve dependencies
    params = read_json(file_path)
    return params



def read_json(file_path):    # Read json file, return as dict
    try:
        with open(file_path, 'r') as file:
            file_data = json.load(file)
    except FileNotFoundError:
        print(f"[ERROR] Cannot locate: %s" % (file_path))
        raise
    return file_data


def list_to_string(from_list, separator=', '):
    out_str = ""
    first_time = True
    for item in from_list:
        if first_time:
            out_str = str(item)
            first_time = False
        else:
            out_str = "%s%s%s" % (out_str, separator, item)
    return out_str

def remove_double_quotes(text):
    text = str(text).replace('"','')
    return text


def read_xml_file(file_path, element_name, attribute_name, attribute_name_2):
    with open(file_path, 'r') as f:
        data = f.read()
    Bs_data_con = BeautifulSoup(data, "xml")
    connections = Bs_data_con.find_all(element_name)
    empty=[]
    for x in range(len(connections)):
        empty.append(str(connections[x]))
    from_=[]
    to_=[]
    for x in range(len(empty)):
        root = ET.fromstring(empty[x])
        from_.append(root.attrib.get(attribute_name))
        to_.append(root.attrib.get(attribute_name_2))
    from_db=pd.DataFrame(from_)
    to_db=pd.DataFrame(to_)
    return from_db, to_db