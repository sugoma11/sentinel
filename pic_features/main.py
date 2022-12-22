import os
import cv2
from modules.create_folder import create_folder
from modules.get_coords import define_coor
from modules.picture import Picture

file_list = os.listdir('raw')
first_pic = cv2.imread(f'raw/{file_list[0]}')

port_name = create_folder()
# defining coords of our regions of interest
"""
Left mouse click - add dot
Right mouse click - remove dot
"Z" button - switch to outer zone
"X" button - switch back to inner zone
"ESC" button - to exit and continue
"""
port_points, out_points = define_coor(first_pic)
print(port_points)

for file in file_list:
    raw = cv2.imread(f'raw/{file}')
    pic = Picture(raw, port_points, out_points)
    pic.plotter()
    pic.add_text()
    pic.aver_split()
    pic.dot_edges(port_points, out_points)
    pic.bitwise()
    pic.aver_split()
    # pic.raw now is not raw, its processed
    cv2.imwrite(f'featured/{port_name}/{file}.jpg', pic.raw)
    os.remove(f'raw/{file}')
