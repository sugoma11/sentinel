import os.path


def create_folder():
    """Creating folder to processed data
    :return: port_name (str): name of port to name directory with processed images"""
    print('Enter port name: ')
    port_name = input()
    os.mkdir(f'featured/{port_name}')

    # temp directory for image with matplotlib graphics
    if not (os.path.exists(f'temp')):
        os.mkdir(f'temp')

    return port_name
