import os.path


def create_folder():
    print('Enter port name: ')
    port_name = input()
    os.mkdir(f'featured/{port_name}')

    if not (os.path.exists(f'temp')):
        os.mkdir(f'temp')

    return port_name
