import os
import cv2
from modules.create_folder import create_folder
from modules.get_coords import define_coor
from modules.get_coords import crop_port
from modules.create_distr import plotter
from modules.add_text import add_text
from modules.area import Area
from modules.add_moments import add_moments
from modules.bitwise_pic import bitwise
from modules.draw_edges import dot_edges


port = Area()
out_port = Area()

file_list = os.listdir('raw')
first_pic = cv2.imread(f'raw/{file_list[0]}')

port_name = create_folder()
port_points, out_points = define_coor(first_pic)

for file in file_list:
    raw = cv2.imread(f'raw/{file}')
    port.image = crop_port(raw, *port_points)
    out_port.image = crop_port(raw, *out_points)
    add_moments(port, out_port)
    plotter(port.image, out_port.image)
    add_text(raw, port, out_port)
    raw = bitwise(raw, port, out_port)
    raw = dot_edges(raw, port_points, out_points)
    cv2.imwrite(f'featured/{port_name}/{file}.jpg', raw)
    os.remove(f'raw/{file}')

