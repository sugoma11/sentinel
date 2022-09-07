import os.path


def create_folder():
    print('Enter port name: ')
    port_name = input()
    os.mkdir(f'C:/Users/gradu/Desktop/python/sentinel_remastered/pic_features/featured/{port_name}')

    if not (os.path.exists(f'C:/Users/gradu/Desktop/python/sentinel_remastered/pic_features/temp')):
        os.mkdir(f'C:/Users/gradu/Desktop/python/sentinel_remastered/pic_features/temp')

    return port_name
