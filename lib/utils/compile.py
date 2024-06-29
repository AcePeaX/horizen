import os

current_folder = os.path.dirname(os.path.realpath(__file__))

def compileFolder(folder_name):
    """
    Compile all the files withing a folder into a class
    Returns an array of text

    Parameters
    ----------
    folder_name: str
        The name of the folder to scrape (within assets folder)
    """
    folder_path = os.path.abspath(os.path.join(current_folder,'..','..','assets',folder_name))
    content = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path,file)) as f:
                content.append("".join(f.readlines()))
    return content
            


if __name__ == '__main__':
    folder = input('Type in the folder name : ')
    result = compileFolder(folder)
    with open(os.path.abspath(os.path.join(current_folder,'..','..','logs',folder+'.txt')), "w") as f:
        f.write("\n\n".join(result))
        f.close()